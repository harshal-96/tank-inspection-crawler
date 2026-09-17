# Self-Correcting Production Cell

A robotic welding cell that measures every part before it welds, judges its own work
afterwards, and escalates to a supervisor agent when the line starts to drift.

Built for the Nebius x NVIDIA Global AI Hackathon, Physical AI track.

> Status: in development. Results tables below are filled in from measured runs only.
> Anything not yet measured is marked TBD rather than estimated.

## The problem

A robot welds where it was taught to weld. A part does not always arrive where it was
taught to be. Sheet metal is cut flat and its sides are folded up, and the press leaves
residual springback and camber on every fold, so the real seam (the line where two
folded edges meet) sits away from the taught path and the weld lands off target. Over
fifty parts measured here the taught path misses by 23 mm on average and 51 mm at worst.

Worse, the seam is not merely displaced, it is bent. It bows off the straight line
between its own ends, so a program that locates the part perfectly and then drives a
straight path still walks off the joint. The usual answers are tighter tolerances
upstream or a skilled operator watching the line. Both are expensive, and neither
scales, and neither straightens a seam that the press has curved.

This project takes the other route: assume the part will move, and measure it.

## What it does

**1. Track the seam, geometrically.** A depth camera looks at the part. The joint is
the line where two folded edges meet, so the cell fits a line to each face and
intersects them, and then walks that measurement up the joint station by station. That
gives the weld path in millimetres directly, with no pixel-to-millimetre calibration to
fit, no labelled images, and nothing to retrain when the part changes, because two
folded edges meeting is still two folded edges meeting.

Tracking the whole length rather than locating a point is the part that matters. The
seam is not merely displaced, it is bent: the press leaves it bowed off the straight
line between its own ends by 0.39 mm on average and 0.94 mm at worst. A program that
locates the part perfectly and then drives a straight path still walks off the joint.

**2. Decide whether to believe it, in the cloud.** One look is not evidence. The
measurement comes with signals a real cell has: the angle between the fitted faces, how
evenly the two faces were seen, the fit residual on each, how much of the joint was
covered. An NVIDIA Nemotron model reads those and decides whether the measurement can
be trusted or whether the part should be presented differently and measured again.

What this catches, measured: across fifty parts the worst individual views came back
seventy millimetres out, and every one of them was the part turned forty five degrees,
where the second face goes nearly edge-on and the fitter locks onto a locating pad
instead. Those views do not hide it. They arrive carrying an inlier balance of 0.10 and
a forty five degree included angle where a square corner should read ninety, and the
model reads exactly those signals. Of eighty nine views taken, sixteen were badly wrong
or refused outright, and all sixteen were kept out of the fused answer.

**3. Send each decision to the model that is actually right about it.** Four Nemotron
sizes are served on Token Factory, and they are not interchangeable. Which one a
decision goes to is worked out from signals already in hand, so the routing costs
arithmetic rather than another model call. What the routing should do turned out to be
the opposite of the obvious answer, and the numbers are under Validation.

**4. Choose the weld to suit the joint that is there.** Tracking says where to go; it
does not say how to weld. The gap between the two folded edges is not constant along
the seam, and a schedule that lays a sound weld on a tight fit blows through a gap
twice as wide. The adjustment is counterintuitive: current and voltage come DOWN as the
joint opens, because arc force pushes a puddle through an open root, and the metal
needed to fill the gap is bought by slowing the torch. Past what one pass can bridge
the cell refuses and says the press fit-up needs correcting, rather than laying metal
into a hole and calling it a weld.

**5. Keep looking while it welds.** Everything measured before the arc strikes describes
a cold part. Weld metal contracts as it cools and rotates both sides about their fold
lines, so welding the bottom of a seam moves the top of it long before the torch gets
there. Calibrated against published distortion figures, the corner walks 1.8 mm over one
weld. The sensor leads the arc by 170 mm, and measuring once at that lead costs 0.75 mm,
which is as large as the entire tracking accuracy. That is the arithmetic that says the
cell has to keep looking rather than measure once.

**6. Inspect the weld, in the cloud.** After welding, a second camera photographs the
bead and a vision model returns a verdict with its reasoning: pass, or a named defect.
Defects are deliberately injected in simulation, so every verdict is scored against a
known answer rather than admired.

**7. Supervise, and diagnose which machine to go and touch.** A Nemotron model runs as a
cell supervisor with tools over the production database. Spotting that something is
drifting is the easy half; the half worth money is saying whether it is the press or the
fixture, because they are fixed at opposite ends of the shop. They look alike in a
summary number and quite different in the stream: a fixture moves the whole seam
together, while a press moves the top of the joint and leaves the bottom where it was,
because a bend angle is applied at a fold and multiplied by height. That discriminator
is free to a cell that tracks the whole seam and invisible to one that locates a point.

The supervisor can also search outside the cell. It measures this part; it does not know
what a code permits or what a filler is rated for, and a model's recollection of a
standard is the least trustworthy thing in the building. Those questions are searched
and answered with sources attached, so an operator can check them.

The division is deliberate. Geometry does what geometry is good at, which is turning
surfaces into millimetres. The models do what geometry cannot: judge whether a
measurement deserves trust, judge whether a weld is sound, diagnose a machine from a
run of parts, and explain any of it to the person responsible for the line.

## How Nebius and NVIDIA are used

| Layer | Platform | Model | Runs today |
| --- | --- | --- | --- |
| View planning, inside the control loop | Nebius Token Factory | Nemotron 3 Nano 30B-A3B | yes |
| Shift-level diagnosis, press vs fixture | Nebius Token Factory | Nemotron 3 Super 120B-A12B | yes |
| Cell supervisor agent, with tool calling | Nebius Token Factory | Nemotron 3.5 Lightning | yes |
| Post-weld inspection | Nebius Token Factory | google/gemma-3-27b-it | yes |
| Looking up standards the cell cannot measure | Tavily | search with sources | yes |

Four NVIDIA Nemotron sizes are served on Token Factory and this project uses three of
them, for different jobs, because measuring them showed they are not interchangeable.
Nothing in this table is aspirational.

The inspection model is not an NVIDIA one, and that is a constraint rather than a
preference. Token Factory serves 24 models; four are Nemotron and all four are
text-only. Exactly one is vision-capable, so the post-weld verdict has one option.
Reasoning about the measurement, which is text, runs on Nemotron; reasoning about the
photograph runs on the only model that can see.

Cosmos is not served on Token Factory at all. Reaching Cosmos Reason or Cosmos Transfer
means self-hosting on Nebius AI Cloud, which is a different platform and a larger piece
of work. It is the obvious next step and no result from it is claimed here.

### One model size does not fit one cell

The obvious architecture is a ladder: a cheap model on routine calls, a large one when
the evidence conflicts. That is what was built, and measuring it took it apart.

Scored against ground truth rather than against the largest model's opinion, on 39
decision points where the simulation knows the answer:

| Model | Correct | Robot moves wasted |
| --- | --- | --- |
| Nemotron 3 Nano 30B | **34 of 39** | 5 |
| Nemotron 3.5 Lightning | 1 of 39 | 38 |
| Nemotron 3 Super 120B | 0 of 39 | 39 |
| Nemotron 3 Ultra 550B | 21 of 39 | 18 |

The smallest model in the family is the most correct one for this decision. The larger
ones are not more careful; they are wrong in a direction that costs robot moves, and a
look is a robot move rather than a line in a bill. Two things make that true, and both
are properties of the cell rather than of the models: accepted views are fused with a
median, so one bad view is outvoted before any model is consulted, and there was nothing
for extra caution to find.

The first version of this evaluation compared each tier against the largest model and
reported that the small one agreed with it zero percent of the time. That was a broken
measurement. On a clean single view the small model says accept and the largest says
look again, and against ground truth accepting is right. Treating the big model as the
reference measured conformity, not quality.

Where the sizes DO separate is the other job. Asked to read thirty parts of measurements
and say whether a press or a fixture is drifting, the small models answer "press" every
time regardless of the data, and do not improve when the signal is made clearer. The
large ones read it:

| Model | Marginal drift | Clearer drift |
| --- | --- | --- |
| Nano, Lightning | 4 of 12, answered "press" 12 of 12 | 4 of 12, unchanged |
| Super, Ultra | 8 of 12 | 10 of 12 |

So the routing that survives is by task, not by evidence difficulty. A fast judgement
over a handful of numbers goes to the smallest model at about 1.4 seconds. Trend
reasoning across a run of parts goes to a large one. Sending the second job to the small
model produces a confident answer that is the same every time.

### Why there is no detector to harden

The usual sim-to-real problem is that a detector trained on rendered images is good
at rendered images and little else. This project does not have that problem, because
it does not have a trained detector. Finding the seam is geometry: two faces fitted
from the depth buffer and intersected, walked along the joint. There is nothing to
label, nothing to train, and nothing to retrain when the part changes, because two
folded edges meeting is still two folded edges meeting.

What that buys is a shorter path to a real machine. The input is a depth map, and a
depth map from a real sensor and a depth map from a render differ far less than two
photographs do. The appearance gap that defeats trained detectors is largely not on
this path.

What it costs is that the geometry has to be right, and getting it right was most of
the work. Three faults in the capture each produced confident, wrong millimetres
before they were found: a camera near plane clipping the corner out of every depth
image, an end stop standing exactly on the seam, and a sheet thickness no press could
fold. None of them looked like a bug. All of them looked like data.

The simulation does emit depth, segmentation and surface normals alongside each
render, which are the control inputs a world foundation model wants. That is the seed
for photorealistic variants through Cosmos Transfer, and it is the obvious next step.
It has not been run, so no result from it is claimed.

## Validation

The cell runs in a physics simulation, which means ground truth exists for every part.
Where the cell says the seam is can be compared against where the seam actually is.
Scored over full campaigns rather than demonstrated once.

Fifty parts arrive under physics with randomised offset, and the press leaves residual
springback and camber on every folded side. Every arm reads the same renders of the
same parts.

### Following the seam

The measurement that matters: not where the corner is as a point, but where the seam
runs along its length, taken at twenty stations up each part. A thousand measurements
over fifty parts.

| | Mean | p95 | Worst | Sub-millimetre |
| --- | --- | --- | --- | --- |
| Taught path, no correction | 22.80 mm | 36.37 | 50.91 | 0 of 50 |
| Tracked from depth | **0.67 mm** | 0.83 | **0.89** | **50 of 50** |

Every part tracked, every station, none refused. The worst single station across all
thousand reads 1.05 mm.

A straight path cannot reach this by being aimed better, because the seam itself is not
straight: it bows off the line between its own ends by 0.39 mm on average
and 0.94 mm at worst.

### Locating the corner, and what the routing costs

The narrower question, and where the model sits in the loop.

| | Mean | p95 | Worst | Sub-millimetre |
| --- | --- | --- | --- | --- |
| No correction | 22.41 mm | 36.11 | 50.17 | 0 of 50 |
| One fixed view | 0.91 mm | 1.61 | 1.79 | 26 of 50 |
| Loop on one fixed model | 0.89 mm | 1.61 | 1.79 | 27 of 50 |
| Loop, routed by task | **0.91 mm** | 1.61 | 1.79 | 26 of 50 |

Routing costs 1.00 looks per part against 1.78, which is
44 percent fewer robot moves, and 74 seconds of
model time against 206, for 0.021 mm of accuracy.

The honest reading on the loop itself is that it is not currently earning its cost on
this measure. An earlier run showed a large gap, and that gap closed when three faults
were fixed in the capture: a camera near plane clipping the corner out of every depth
image, an end stop standing on the seam, and a sheet thickness no press could fold.
Those fixes made the single fixed view good. What the loop demonstrably still does is
refuse bad evidence, and what routing does is stop paying a large model to be cautious
about parts that were never in doubt.

### What the arc does to the joint

Calibrated against published distortion figures rather than fitted to look good: about
0.25 degrees of angular distortion and 0.94 mm of transverse shrinkage for a weld this
size, angular distortion rising with heat input and falling with the square of the
thickness.

| | |
| --- | --- |
| Corner walks over one weld | 1.84 mm |
| Residual springback the press leaves | 0.11 deg typical |
| Angular distortion the arc adds | 0.25 deg |
| Cost of measuring once at a 170 mm lead | 0.75 mm |

The arc moves the joint by more than the press did, and it does it after the
measurement was taken. The last row is why the sensor cannot look once.

### Diagnosing which machine to touch

Three scenarios where the answer is known: a press drifting, a fixture drifting, and
nothing wrong at all. The control matters, because a diagnostician that always finds
something is not diagnosing, and a control run has to be checked for flatness before it
can be used, since ordinary scatter produces trends by chance.

Results are in the routing section above. The short version: this task needs a large
model, and the small ones answer "press" every time.

### Reproducing all of it

```bash
python scripts/run_seam_campaign.py --manifest data/cap50/manifest.json
python scripts/run_campaign.py --manifest data/cap50/manifest.json --no-post --routed
python scripts/eval_router_truth.py --campaign data/campaign50.jsonl
python scripts/eval_drift.py --parts 30 --runs 4
python scripts/draw_result_chart.py --campaign data/seam_campaign.jsonl
```

Seam finding is geometric: two faces fitted from depth and intersected, walked along
the joint. There is no trained model in that path, nothing to label and nothing to
retrain when the part changes.

## Architecture

```
  IN THE CELL                             CLOUD (Nebius Token Factory)
  -----------                             ----------------------------
  part arrives, settles under physics
        |
        v
  depth camera, one fixed pose
        |
        v
  locate the part, then track the    --->   Nemotron Nano: is this view worth
  seam station by station up the             believing, or present the part
  joint: two faces fitted from                and look again?
  depth and intersected                                |
     ^                    |                            |
     |                    |                            |
     +-- part re-presented about the joint axis <------+
                          |
                    fuse accepted views
                          |
                          v
     gap at every station --> weld schedule per station
                          |   arc down, travel slowed, or refuse the fit-up
                          v
                    weld the measured path
                          |
     the arc rotates both sides as it goes, so the joint keeps moving:
     the sensor leads the arc and keeps measuring rather than trusting
     the cold reading
                          |
                          v
  post-weld capture  -------------->      vision model -> pass, or a named
                          |                               defect, with reasoning
                          v
  FastAPI service + SQLite  <---------->  Nemotron Super: across a run of parts,
        |                                  is the PRESS drifting or the FIXTURE?
        |                                  a press tilts the seam, a fixture
        +--> operator dashboard            shifts it
        +--> Nemotron supervisor, reading the same rows through tools,
             and searching outside the cell for anything it cannot measure

  Ground truth exists for every part, so every number above is scored, not shown.
```

Every measurement, verdict and decision lands in one database. The dashboard and the
agent read the same rows, so what the operator sees and what the agent reasons over
cannot drift apart.

## Repository layout

```
cell/       cell geometry, press forming and drift, weld thermal distortion,
            adaptive weld parameters, depth seam tracking, Isaac simulation
brain/      Nebius client, model router, view planner, inspection prompt,
            supervisor agent and its tools, outside lookup
service/    FastAPI service, database schema, API
dashboard/  operator web UI
scripts/    campaign runners, evaluations, figures, cell visualisation
docs/       platform notes and measured results
tests/      unit tests
```

The pieces that carry the argument:

| File | What it settles |
| --- | --- |
| `cell/seam_track.py` | follows the seam along its length instead of fitting it straight |
| `cell/forming.py` | what a press in spec still leaves on a folded side, and what a drifting one leaves |
| `cell/thermal.py` | what the arc does to the joint while it is welding it |
| `cell/weld_params.py` | choosing the weld to suit the gap that is actually there |
| `brain/router.py` | which Nemotron each decision goes to, and why that is not the obvious answer |
| `scripts/eval_router_truth.py` | scores every model size against ground truth rather than against each other |
| `scripts/eval_drift.py` | press or fixture, with a control |

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

Run the service and open the dashboard at http://127.0.0.1:8000:

```bash
uvicorn service.app:app --reload
```

It starts empty. Fill it from a capture, which takes a few minutes and needs no GPU
because the depth buffers are already on disk:

```bash
python scripts/run_seam_campaign.py --manifest data/cap50/manifest.json
python scripts/run_campaign.py --manifest data/cap50/manifest.json --routed
python scripts/run_inspection.py --manifest data/cap50/manifest.json
```

The first posts a tracked seam and a weld schedule per part, the second the
look-and-judge decisions with the model size that made each one, the third the
post-weld verdicts scored against the defects that were injected.

The dashboard then shows the seam itself: the tracked path against the straight line
a taught program drives, the gap at every station, the heat the torch is putting in,
and which Nemotron answered each decision and why.

The simulation half requires Isaac Sim 6.0 and is run with its bundled interpreter. See
`docs/simulation.md`. Capture in batches of ten rather than one run of fifty; the
Replicator graph grows as parts times frames and a single large capture stalls in
graph construction. `scripts/merge_manifests.py` stitches the batches.

## Prior work

The joint-finding and correction concept was validated by the author in an earlier
project, published at
[vision-corrected-welding](https://github.com/harshal-96/vision-corrected-welding).
Everything in this repository was built during the hackathon submission period: the
press forming and drift model, seam tracking along the length of the joint, the weld
thermal distortion model, adaptive weld parameters from the measured gap, the view
planner that puts a reasoning model inside the control loop, the task-based model
router and the evaluations that overturned its first design, the press-versus-fixture
diagnosis, the post-weld inspection station and defect injection, the cloud inspection
model, the supervisor agent with its tools and outside lookup, the database and
service, and the dashboard. The earlier work is the starting point, not the
submission.

## License

Apache License 2.0. See [LICENSE](LICENSE).
