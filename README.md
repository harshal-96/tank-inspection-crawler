# Self-Correcting Production Cell

A robotic welding cell that measures every part before it welds, judges its own work
afterwards, and escalates to a supervisor agent when the line starts to drift.

Built for the Nebius x NVIDIA Global AI Hackathon, Physical AI track.

> Status: in development. Results tables below are filled in from measured runs only.
> Anything not yet measured is marked TBD rather than estimated.

## The problem

A robot welds where it was taught to weld. A part does not always arrive where it was
taught to be. Sheet metal carries camber, springback and fit-up variation, so the real
joint (the line where two panels meet) sits a few millimetres away from the taught path,
and the weld lands off target. The usual answers are tighter tolerances upstream or a
skilled operator watching the line. Both are expensive, and neither scales.

This project takes the other route: assume the part will move, and measure it.

## What it does

Three things happen to every part.

**1. Measure and correct, at the edge.** A fixed camera photographs the clamped part. A
detector running locally on an optimised inference engine finds the actual joint, a
calibrated mapping converts pixels to robot coordinates, and the robot retargets before
it strikes an arc. This runs on the machine, in milliseconds, because a cell cannot wait
on a network round trip to decide where to weld.

**2. Inspect, in the cloud.** After the weld, a second camera photographs the bead. An
NVIDIA Nemotron vision model running on Nebius Token Factory returns a verdict with its
reasoning: pass, or a named defect. Quality judgement is a slower, harder, more
linguistic problem than localisation, so it goes where the large models are.

**3. Supervise.** An NVIDIA Nemotron reasoning model runs as a cell supervisor with tools
over the production database. It reads the per-part measurement stream, spots drift that
no single part would reveal, decides accept, rework or recalibrate, writes shift reports,
and answers the operator in plain language.

That split is the point. Millimetre correction belongs at the edge. Judgement,
pattern-finding across a shift, and explanation belong in the cloud.

## How Nebius and NVIDIA are used

| Layer | Platform | Model |
| --- | --- | --- |
| Post-weld inspection | Nebius Token Factory | NVIDIA Nemotron vision model |
| Cell supervisor agent | Nebius Token Factory | NVIDIA Nemotron reasoning model, with tool calling |
| Detector training | Nebius AI Cloud GPUs | Detector fine-tuned on synthetic data |

## Validation

The cell runs in a physics simulation, which means ground truth exists for every part.
Where the detector says the joint is can be compared against where the joint actually is,
and a defect that was deliberately injected into a weld bead can be compared against the
verdict the vision model returned. Both are scored over full campaigns rather than
demonstrated once.

| Metric | Result |
| --- | --- |
| Correction error, mean | TBD |
| Correction error, p95 | TBD |
| Inspection accuracy | TBD |
| Inspection confusion matrix | TBD |
| Edge inference latency | TBD |

A second station runs the same detector on a real camera and a real metal part, so the
perception path is shown working outside simulation as well.

## Architecture

```
  EDGE (local)                          CLOUD (Nebius Token Factory)
  ------------                          ----------------------------
  camera -> detector -> correction        Nemotron vision  -> inspection verdict
     |                      |             Nemotron reason  -> supervisor decisions
     |                      v                     ^
     |              robot retargets               |
     v                                            |
  post-weld capture ------------------------------+
     |
     v
  FastAPI service + database  <->  operator dashboard
```

Every measurement, verdict and decision lands in one database. The dashboard and the
agent read the same rows, so what the operator sees and what the agent reasons over
cannot drift apart.

## Repository layout

```
cell/       edge perception: geometry, calibration, detector, simulation, bench station
brain/      Nebius clients, the inspection prompt, the supervisor agent and its tools
service/    FastAPI service, database schema, API
dashboard/  operator web UI
scripts/    setup, campaign runners, report generation
docs/       platform notes and measured results
tests/      unit tests
```

## Setup

Requires Python 3.12 and a Nebius Token Factory API key.

```bash
git clone <this-repo>
cd self-correcting-cell
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # Linux and macOS
pip install -r requirements.txt
```

Copy the environment template and fill in your key:

```bash
cp .env.example .env
```

Verify cloud access before anything else:

```bash
python scripts/hello_nemotron.py
```

Run the service and open the dashboard:

```bash
uvicorn service.app:app --reload
```

The simulation half requires Isaac Sim 6.0 and is run with its bundled interpreter. See
`docs/simulation.md`.

## Prior work

The joint-finding and correction concept was validated by the author in an earlier
project, published at
[vision-corrected-welding](https://github.com/harshal-96/vision-corrected-welding).
Everything in this repository was built during the hackathon submission period: the
post-weld inspection station and defect injection, the cloud inspection model, the
supervisor agent and its tool interface, the production database and service, the
dashboard, and the bench station. The earlier work is the starting point, not the
submission.

## License

Apache License 2.0. See [LICENSE](LICENSE).
