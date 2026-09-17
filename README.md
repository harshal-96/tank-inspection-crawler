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

**1. Measure the joint, geometrically.** A depth camera looks at the clamped part. A
fillet joint is the line where two panel faces meet, so the cell fits a plane to each
face and intersects them. That gives the weld path in millimetres directly, with no
pixel-to-millimetre calibration to fit, no labelled images, and nothing to retrain when
the part changes, because every fillet joint is still two planes meeting. This is the
principle a laser seam tracker works on.

**2. Decide whether to believe it, in the cloud.** One look is not evidence. The
measurement comes with signals a real cell has: the angle between the fitted faces, how
evenly the two faces were seen, the fit residual on each, how much of the joint was
covered. An NVIDIA Nemotron model reads those and decides whether the measurement can be
trusted or whether the part should be presented differently and measured again. The robot
re-presents, the cell measures again, and the accepted views are fused.

This is the part that matters. A single view can look perfectly healthy and still be
twenty millimetres out, and nothing inside that view will say so. Two views disagreeing
say so at once.

**3. Inspect the weld, in the cloud.** After welding, a second camera photographs the
bead and a vision model returns a verdict with its reasoning: pass, or a named defect.
Defects are deliberately injected in simulation, so every verdict is scored against a
known answer rather than admired.

**4. Supervise.** A Nemotron reasoning model runs as a cell supervisor with tools over
the production database. It reads the per-part stream, spots drift that no single part
would reveal, decides accept, rework or recalibrate, writes shift reports, and answers
the operator in plain language. It records recommendations; it does not move the cell.

The division is deliberate. Geometry does what geometry is good at, which is turning
surfaces into millimetres. The models do what geometry cannot: judge whether a
measurement deserves trust, judge whether a weld is sound, and explain either to the
person responsible for the line.

## How Nebius and NVIDIA are used

| Layer | Platform | Model |
| --- | --- | --- |
| Cell supervisor agent | Nebius Token Factory | NVIDIA Nemotron, with tool calling |
| Post-weld inspection | Nebius Serverless Endpoints | NVIDIA Cosmos Reason, self-hosted |
| Synthetic data hardening | Nebius Serverless Jobs | NVIDIA Cosmos Transfer |
| Detector training | Nebius Serverless Jobs | Detector trained on hardened synthetic data |
| Detector serving | Nebius Serverless Endpoints | The trained detector behind HTTPS |

### Closing the gap between simulation and reality

A detector trained on rendered images is good at rendered images. That is the
standing weakness of synthetic training data, and the reason most simulation work
never reaches a real machine.

The simulation emits more than a photograph. Every frame also carries a depth map,
a segmentation mask and surface normals, computed from the same render. Those are
exactly the control inputs a world foundation model needs, so each rendered weld
becomes the seed for a set of photorealistic variants that keep the geometry but
change the appearance.

The detector is then trained on the hardened set, and both detectors, the one
trained on raw renders and the one trained on hardened renders, are measured on
photographs of a real part taken with a real camera. The difference between those
two numbers is the result this project is built to produce.

## Validation

The cell runs in a physics simulation, which means ground truth exists for every part.
Where the detector says the joint is can be compared against where the joint actually is,
and a defect that was deliberately injected into a weld bead can be compared against the
verdict the vision model returned. Both are scored over full campaigns rather than
demonstrated once.

Joint position error against the true joint, over 12 parts arriving under physics
with randomised offset. Both arms read the same renders of the same parts.

| | Mean | p95 | Max | Sub-millimetre |
| --- | --- | --- | --- | --- |
| No correction | 23.17 mm | 40.72 | 40.72 | 0 of 12 |
| One fixed view, camera well placed | 2.25 mm | 19.92 | 19.92 | 11 of 12 |
| One fixed view, camera poorly placed | measurement failed on 11 of 12 | | 64.73 | 0 of 12 |
| Look, judge, look again | **0.64 mm** | 0.67 | **0.67** | **12 of 12** |

The honest reading is that the loop does not make a good measurement better. The
median barely moves. What it removes is the tail: one part in twelve presented a
face nearly edge-on to the fixed camera and came back 19.9 mm out, with nothing in
that single view to say so. Two views disagreeing by 20 mm says so immediately. On
a line, the tail is what scraps parts.

It also removes the dependence on getting the camera placement right. Moved to an
angle that suits this part family badly, the fixed view stops working altogether
while the loop still lands every part inside 0.7 mm.

Joint finding is geometric: two planes fitted to the panel faces and intersected.
There is no trained model in that path, nothing to label and nothing to retrain
when the part changes, because every fillet joint is still two planes meeting.

A second station runs the same detector on a real camera and a real metal part, so the
perception path is shown working outside simulation as well.

## Architecture

```
  TRAINING (Nebius Serverless Jobs)
  ---------------------------------
  simulation render + depth + segmentation
        |
        v  Cosmos Transfer            photorealistic variants
        |
        v  training job               hardened detector
        |
        v  Serverless Endpoint        served over HTTPS
        |
  ======|=====================================================
        v
  EDGE (in the cell)                    CLOUD (Nebius)
  ------------------                    --------------
  depth -> plane fit -> weld path  --->   Nemotron: trust this view,
     ^                    |                         or look again?
     |                    |                              |
     +-- robot re-presents the part <--------------------+
                          |
                    fuse, then weld
                          |
                          v
  post-weld capture  -------------->      Cosmos Reason -> inspection verdict
                                          Nemotron      -> supervisor decisions
     |
     v
  FastAPI service + database  <->  operator dashboard

  Both detectors are scored on photographs of a real part from a real camera.
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
