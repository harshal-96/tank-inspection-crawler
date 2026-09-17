"""Train a small model on the drift diagnosis, on Nebius Token Factory.

Why this exists. The routing study found one question the small model cannot
answer: given a run of parts, name which machine is going out of adjustment. Nano
scores 4 of 12 by saying "press" every time. Only the 120B reads the evidence, and
a 120B is not something a cell asks once a part.

Why fine-tuning is the right answer rather than a bigger prompt. The data is not
scarce. The cell is a simulation with a forming model in it, so runs with a known
fault arrive already labelled and cost nothing. That is the Physical AI trade in
miniature: simulate, label for free, train, and put the result where the work is.

What it does NOT do. Token Factory does not offer Nemotron as a fine-tuning base,
so the student is the smallest capable open model on the supported list. The
Nemotron sizes stay where the measurements put them: in the loop, and as the
baseline this result is measured against.

    python scripts/finetune_drift.py --launch
    python scripts/finetune_drift.py --watch <job_id>
    python scripts/finetune_drift.py --fetch <job_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import nebius  # noqa: E402

# Small on purpose. The claim being tested is that the shift-level question can
# come off the cloud entirely, and a model that needs a datacentre to answer it
# has not moved the question anywhere.
BASE_MODEL = "Qwen/Qwen3-1.7B"

HYPERPARAMETERS = {
    "batch_size": 8,
    "learning_rate": 1e-4,
    "n_epochs": 3,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "lora": True,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "packing": True,
    "max_grad_norm": 1.0,
    "context_length": 8192,
}

TERMINAL = ("succeeded", "failed", "cancelled")
STATE = Path("data/finetune/job.json")


def client():
    return nebius.client()


def launch(args) -> int:
    api = client()
    train_path = Path(args.train)
    val_path = Path(args.val)
    for path in (train_path, val_path):
        if not path.is_file():
            raise SystemExit(f"missing {path}; run scripts/make_drift_dataset.py first")

    print(f"uploading {train_path} and {val_path}")
    training_file = api.files.create(file=train_path.open("rb"), purpose="fine-tune")
    validation_file = api.files.create(file=val_path.open("rb"), purpose="fine-tune")
    print(f"  training   {training_file.id}")
    print(f"  validation {validation_file.id}")

    job = api.fine_tuning.jobs.create(
        model=args.base,
        training_file=training_file.id,
        validation_file=validation_file.id,
        suffix=args.suffix,
        seed=42,
        hyperparameters=HYPERPARAMETERS,
    )
    print(f"\njob {job.id}  base {args.base}  status {job.status}")

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({
        "job_id": job.id,
        "base_model": args.base,
        "training_file": training_file.id,
        "validation_file": validation_file.id,
        "hyperparameters": HYPERPARAMETERS,
        "launched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")
    print(f"wrote {STATE}")
    return 0


def job_id_from(args) -> str:
    if args.job:
        return args.job
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))["job_id"]
    raise SystemExit("no job id given and no data/finetune/job.json to read one from")


def watch(args) -> int:
    api = client()
    job_id = job_id_from(args)
    seen: set[str] = set()
    while True:
        job = api.fine_tuning.jobs.retrieve(job_id)
        try:
            for event in reversed(api.fine_tuning.jobs.list_events(job_id, limit=20).data):
                key = f"{event.created_at}:{event.message}"
                if key not in seen:
                    seen.add(key)
                    print(f"  {event.message}")
        except Exception:  # noqa: BLE001 - events are a convenience, not the status
            pass
        step = getattr(job, "trained_steps", None) or 0
        total = getattr(job, "total_steps", None) or "?"
        done = f"{step}/{total}"
        print(f"[{time.strftime('%H:%M:%S')}] {job.status}  steps {done}")
        if job.status in TERMINAL:
            if job.status != "succeeded":
                print(f"\nfailed: {getattr(job, 'error', None)}")
                return 1
            print(f"\nsucceeded. fine-tuned model: {getattr(job, 'fine_tuned_model', None)}")
            return 0
        if args.once:
            return 0
        time.sleep(args.interval)


def to_peft_layout(adapter_dir: Path) -> None:
    """Rewrite the exported adapter into the layout peft's loader expects.

    Measured, not assumed. Token Factory exports its LoRA tensors named from the
    base model's own module tree:

        model.layers.0.mlp.down_proj.lora_A.weight

    peft looks for them under the wrapper it puts around that tree:

        base_model.model.model.layers.0.mlp.down_proj.lora_A.weight

    Loading one as the other does not fail. peft warns about missing keys, leaves
    every LoRA weight at its initial value and hands back a model that is
    numerically the base model. The first run of this evaluation scored 0 of 12
    and took ten seconds a case, which is exactly what the untrained 1.7B does,
    and the only reason it was caught is that the warning was read rather than
    scrolled past. A silent no-op is the worst kind of bug in an evaluation,
    because it looks like a result.

    The config needs one field too: the export leaves task_type null, and peft
    needs CAUSAL_LM to wire the adapter to a causal language model.
    """
    import json as _json

    from safetensors.torch import load_file, save_file

    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.is_file():
        return
    tensors = load_file(str(weights_path))
    if not tensors or next(iter(tensors)).startswith("base_model."):
        print("  adapter already in peft layout")
        return

    remapped = {f"base_model.model.{k}": v for k, v in tensors.items()}
    save_file(remapped, str(weights_path))
    print(f"  renamed {len(remapped)} tensors into peft's layout")

    config_path = adapter_dir / "adapter_config.json"
    config = _json.loads(config_path.read_text(encoding="utf-8"))
    if not config.get("task_type"):
        config["task_type"] = "CAUSAL_LM"
        config_path.write_text(_json.dumps(config, indent=2), encoding="utf-8")
        print("  set task_type to CAUSAL_LM")


def fetch(args) -> int:
    """Pull the adapter down, because the cell is where it has to run."""
    api = client()
    job_id = job_id_from(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    checkpoints = api.fine_tuning.jobs.checkpoints.list(job_id).data
    if not checkpoints:
        raise SystemExit("no checkpoints on this job yet")
    for checkpoint in checkpoints:
        print(f"checkpoint {checkpoint.id}  step {checkpoint.step_number}  "
              f"metrics {getattr(checkpoint, 'metrics', None)}")
    latest = max(checkpoints, key=lambda c: c.step_number)
    print(f"\ntaking {latest.id} (step {latest.step_number})")

    target = out / "adapter"
    target.mkdir(parents=True, exist_ok=True)
    for file_id in latest.result_files:
        meta = api.files.retrieve(file_id)
        name = os.path.basename(meta.filename)
        api.files.content(file_id).write_to_file(str(target / name))
        print(f"  {name}")

    to_peft_layout(target)
    print(f"\nwrote {target}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--watch", dest="watch_job", action="store_true")
    parser.add_argument("--fetch", dest="fetch_job", action="store_true")
    parser.add_argument("--job", default=None, help="job id, or read data/finetune/job.json")
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--suffix", default="drift-diagnosis")
    parser.add_argument("--train", default="data/finetune/drift_train.jsonl")
    parser.add_argument("--val", default="data/finetune/drift_val.jsonl")
    parser.add_argument("--out", default="data/finetune")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--once", action="store_true", help="print status once and stop")
    args = parser.parse_args()

    if args.launch:
        return launch(args)
    if args.fetch_job:
        return fetch(args)
    if args.watch_job or args.job or STATE.is_file():
        return watch(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
