# Nebius platform notes

Verified 17 September 2026 against the live catalog and the published OpenAPI spec.
Recorded here because two of these facts contradict Nebius's own documentation and
blog posts, and a wrong model ID costs a day.

## Two products, two credentials

Nebius ships two things that are easy to confuse.

| | Token Factory | AI Cloud |
| --- | --- | --- |
| What | Hosted model inference | GPU VMs, Slurm, Kubernetes |
| Console | tokenfactory.nebius.com | console.nebius.com |
| Auth | API key, `Authorization: Bearer` | IAM access token, 12 hour lifetime |
| Identifiers | project API key | `user_id`, `org_id` |

An AI Cloud `user_id` and `org_id` will not authenticate against Token Factory. The
inference key has to be created separately inside a Token Factory project.

## Inference endpoint

Token Factory implements the OpenAI wire protocol, so the official SDK works with a
swapped base URL.

```
https://api.tokenfactory.nebius.com/v1/
```

Regional hosts (`eu-north1`, `us-central1`) and the legacy `api.studio.nebius.com` all
resolve and return 401 unauthenticated. The plain host above is the documented one.

## There is no NVIDIA vision model on Token Factory

This is the load-bearing finding for any project that planned on one.

NVIDIA's two multimodal models were withdrawn from serverless inference on
**31 August 2026**:

| Withdrawn | Replacement Nebius offered |
| --- | --- |
| `nvidia/Nemotron-3-Nano-Omni` | `nvidia/Nemotron-3_5-Lightning`, which is text only |
| `nvidia/Cosmos3-Super-Reasoner` | `MiniMaxAI/MiniMax-M3`, which is not an NVIDIA model |

All four NVIDIA models remaining in the catalog are text only. That was measured, not
read from the catalog: on 19 September 2026 `scripts/probe_vision_field.py` put every
model the key reaches, 23 of them, through the inspection path with one clean bead and
one burnt through (`data/vision_field.jsonl`). 15 refused the image outright, all four
NVIDIA models among them. 8 accepted it; 6 of those judged the bead and 2 answered
"uncertain" on both beads (MiniMax-M3 and Kimi K3). Nebius marketing pages
still announce "Nemotron Nano 2 VL in AI Studio"; that is stale. The Hugging Face repo
`nvidia/Nemotron-Nano-12B-v2-VL` does not exist either, the real ones are suffixed
`-BF16`, `-FP8` and `-NVFP4-QAD`.

Serving an NVIDIA vision model on Nebius therefore requires self-hosting: a dedicated
endpoint, custom weights (beta, request only), or vLLM on an AI Cloud GPU.

## Model IDs

The API model ID lives at `flavors[].model_id` in the catalog. It is **not** the
display name and **not** always the Hugging Face repo name. Casing is inconsistent
between entries, so copy them verbatim.

Public, unauthenticated catalog:

```
https://tokenfactory.nebius.com/proxy/inference/private/v1/models_info
```

Serverless and confirmed for this project:

| Role | `model_id` | In / out per 1M | Context | Tools |
| --- | --- | --- | --- | --- |
| Supervisor agent, and the corrective action briefs | `nvidia/Nemotron-3_5-Lightning` | $0.06 / $0.24 | 1M | yes |
| View planning in the cell; measured as the tank's probe planner | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | see the catalog | see the catalog | see the catalog |
| Weld inspection, and the shift diagnosis | `deepseek-ai/DeepSeek-V4.1-Flash` | see the catalog | see the catalog | yes |
| Weld inspection, earlier choices | `MiniMaxAI/MiniMax-M3`, `google/gemma-3-27b-it` | see the catalog | see the catalog | yes |

Gemma was the inspector until the fifty beads were re-run under a prompt that tells the
model the picture is a rendering. Under it MiniMax-M3, DeepSeek V4.1 Flash and Gemma tie
within noise on 50 beads. On 150 beads DeepSeek V4.1 Flash wins the pass-or-defect call
outright, so it is the inspector; the README has the tables.

Nemotron 3 Nano does live work in the cell, deciding whether a view is worth believing
or the part should be presented again. On the tank it was measured as the planner that
picks where the thickness probe goes, and it lost: 9, 10 and 12 of the 28 rejectable
points in three recorded runs, and 11 with the camera's report withheld, against 16 for
a coverage rule given the same budget (`tank/data/survey/probing.json`,
`probing_power.json`, `probing_blind.json`). So the rule drives the probe and the model
result is reported as a negative one. Those runs used the default reasoning setting,
unlike the cell's view planner and brief writer, which set `reasoning_effort` to
`"none"` (see below); the result holds for that setting only.

Twelve of the twenty-four catalog entries carry a `dedicated-endpoint` use case. Whether
that means dedicated-only or merely dedicated-also is undocumented. Every model in the
table above answered serverless calls on a plain project key. Confirm what a given key
can reach with `client.models.list()`, which is what `scripts/hello_nemotron.py` does.

## Request details that matter

- `max_tokens` defaults to **8192** when omitted. Set it explicitly for long agent turns.
- Thinking is controlled by `reasoning_effort`, an enum from `none` through `max`. The
  Nemotron v2 control tokens `/think` and `/no_think` were dropped in the 3 generation
  and do nothing.
- `reasoning_content` is present on streaming deltas but not on the non-streaming
  response schema. Stream if the trace is needed, and strip a stray leading
  `<think>` block defensively.
- Structured output is documented as `extra_body={"guided_json": ...}`, with
  `response_format` accepting only `json_object` and `text`. Measured against
  Nemotron 3.5 Lightning, **neither has any effect**: both were ignored and the
  model still answered in prose. What does work is `reasoning_effort`. The model
  reasons in plain text rather than in a tagged block, so it will spend an entire
  token budget thinking before reaching the answer, and `"low"` is not enough to
  stop it. `"none"` returns the answer directly, in this case cutting 900 tokens of
  preamble down to 86 tokens of correct JSON. Any call that needs parseable output
  should set it.
- Image content parts take a public URL or a base64 data URL. The URL string is capped
  at **13,981,514 characters**, roughly 10 MiB of raw image.

## Rate limits and billing

Limits are dynamic on rolling 15 minute windows: sustained use above 80 percent raises
the ceiling, light use lowers it, capped at 20 times the base. Responses carry
`x-ratelimit-*` headers and a 429 carries `Retry-After`, which should be honoured.

The standard Token Factory trial is **$1, valid 30 days, and requires a card**. The AI
Cloud free trial was **suspended on 13 July 2026**. Budget accordingly, and check which
product any promotional credit actually applies to.

## Guarding against drift

Nebius removed models in June 2026 and again in August 2026 without rerouting requests.
A hardcoded model ID is a liability. `scripts/hello_nemotron.py` checks the configured
IDs against the live catalog before any run.
