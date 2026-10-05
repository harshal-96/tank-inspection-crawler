# Tank Inspection Crawler

A robot that crawls the weld seams inside a steel storage tank, the lines where its steel
plates are welded together, with two instruments: a camera that judges the surface, and an
ultrasonic probe that measures what is left of the plate behind it. The premise this project is built on, and standard API 653 practice, is
that **neither instrument can do the other's job**. The camera cannot find wall loss that
leaves no mark on the face it sees, and here it cannot even see 6 of the 7 places the probe
can reach. The probe cannot judge a weld surface. So the model belongs on the picture, and
arithmetic drives the arm.

What we measured, on a replayable scene built so that only the probe can find the
rejectable wall loss: a vision model, DeepSeek V4.1 Flash, makes the walk-in call right on
23 of 24 frames (a development set; the one miss is a crack the frame before had already
named). For choosing where to probe, NVIDIA Nemotron 3 Nano was never ahead of a short
coverage rule as shipped: the rule found 16 of 28 rejectable points and all 5 hidden
patches, the model 9 to 12 and 4 of 5 in each run. The 16 is a lucky tie-break order. Over
all 5040 orders the rule finds a median of 12, and 13 when shown only what the model sees,
so typically 12 to 13 against the model's 9 to 12, and no run differs from it by a paired
exact test. Hiding the camera from the model showed no change in one run, which on this
rig is largely by construction. Arithmetic drives the arm because on this evidence it is
no worse, and it is deterministic, free, and adds no hours to a tank.

Where NVIDIA models run live: on the welding cell this loop was first built on (below).
Nemotron 3 Nano makes every per-part decision about whether a measurement is good enough
to weld on, 88 of 89 right against ground truth, where the same limits written as
arithmetic get 86; that gap is not one an exact paired test can call (p 0.625). Nemotron
3.5 Lightning drives the supervisor agent and writes the corrective action briefs. In the
tank, Nemotron was measured as a probe planner and is not used. Every hosted model call
runs on Nebius Token Factory; the offline diagnosis model was trained there and runs on
the cell.

Built for the Nebius x NVIDIA Global AI Hackathon, Physical AI track.

Confined space entry is what makes this work dangerous. A tank has to be emptied,
cleaned, ventilated and certified before anyone climbs in, and the inspection itself is a
person with a lamp and a probe standing inside a steel drum. The crawler does not remove
that outage. It works in air, inside a tank already drained for its floor inspection, and
cuts the hours people spend inside it. What it cannot defend, it hands to a person, and it
says so.

## If you have ten minutes

This file is long because every number has its method written next to it. If you are
reading it against a clock, these commands need no API key, no GPU and no download.

```bash
pip install -r requirements.txt
python tank/score_survey.py --replay      # the 24 tank frames, recorded verdicts re-marked
python tank/score_probes.py --baselines   # where to probe: arithmetic, chance and oracle
python tank/significance.py               # what each of those counts actually licenses
python -m pytest -q                       # 383 tests
```

None of these calls a model, even with a Token Factory key set, and none of them writes a
file. Asking a model again is a separate flag on each scorer, `--live`, which is paid and
writes into `tank/data/survey/runs/` by default, not over the recorded results.

The 24 frames are **in this repository**, in `tank/data/survey/frames`, next to the
answer key, which is computed from the scene and not from any model's answer. Open them and
disagree with the scoring if you like.

## The argument: two instruments, and neither is enough

The survey run carries six patches of wall loss. One shows as rust on the surface and a
camera finds it at once; it has **69 percent** of its wall left and passes. The other
five leave no mark on the face the crawler sees; the worst has **34.5 percent** left and
fails. The limit is 60 percent of nominal wall, a screening threshold set for this
benchmark. It is not an API 653 minimum thickness, which is calculated course by course
from the tank's dimensions, fill height and product.

The scene encodes that premise on purpose (`tank/sim/blender_tank.py`): one visible patch
inside limits, five hidden ones outside, placed by hand 0.27 to 0.41 m off the seam. What
it measures is how each planner copes with that, not whether it holds in a given tank. The
loss it stands for is eaten from the far side of the plate: corrosion under insulation on
an insulated or heated shell, soil-side and chime loss, and, the strongest case, floor
plates whose underside nobody can see.

Across the run, 28 of the 168 places the probe can reach fail the limit, all of them on
the five hidden patches and none on the visible one. **20 of those 28 sit under frames the
inspector passed**, and 18 under frames with no visible fault in them at all.
`tank/score_probes.py` prints these counts from the manifest every time it runs.

The geometry leaves the camera very little to say about the probe. A frame covers 0.295 m
along the seam and 0.166 m across it, about ±0.083 m. The probe reaches seven offsets: 0,
±0.15, ±0.30 and ±0.45 m. So the camera images only the on-seam position, and 6 of the 7
offsets are never in any frame. The camera also sits 0.36 m ahead of the probe contact, so
the picture of a station is taken before the probe gets there.

The arm has time for two measurements in seven at each station. That is a stated design
assumption for a single-element spot probe, placed, coupled and read at each point, not a
measured cycle time; a probe that reads continuously as it rolls would take the choice
away. With two in seven, something has to decide where to measure.

## What was measured, and what it licenses

Every result is a count out of a small number, so each one is printed with the bound it
supports rather than left to look better than it is. `tank/significance.py` prints the
inspection bounds and the model runs with their paired tests; `tank/score_probes.py
--baselines` prints the arithmetic rows, chance and the oracle, and `--tiebreaks` the
tie-break ranges; `tank/score_survey.py --replay` shows the inspection marks frame by
frame.

**The inspection.** DeepSeek V4.1 Flash, one recorded verdict a frame, 24 frames.

| | Result | One sided 95 percent exact lower bound |
| --- | --- | --- |
| Is this seam sound, or does somebody have to walk in | **23 of 24**; 1 missed, no false alarms, none declined | 0.817 |
| The same call, counted only when the fault it names is really in the frame | 22 of 24 | 0.760 |
| Finding every fault in the frame, not only the worst | **10 of 15**, none invented | 0.423 |
| Frames with every fault found | 19 of 24 | 0.611 |
| Naming the worst fault | 19 of 24 | 0.611 |
| Distinct faults on the run named in at least one frame that holds them | 4 of 5 | 0.343 |
| Frames holding a fault, flagged | 10 of 11 | 0.636 |
| Frames holding a fault, flagged for a fault really in view | 9 of 11 | 0.530 |
| Clean frames passed | 13 of 13 | 0.794 |

- The miss is station 12: the frame holds the same 64 mm crack that frame 11 had already
  named, and the inspector passed it.
- Calling every frame sound would score 13 of 24 on the first row, because 13 frames hold
  no fault. The pooled count is mostly clean frames, so the last three rows split it by
  class, and the 11 frames that hold a fault are the real test.
- The strict row is lower because frame 1 holds D001, a 1.7 mm undercut against a 0.8 mm
  limit, and was called defect for a sliver of rust, 0.01 of the picture, under the
  counting threshold. The right call for the wrong reason. Frame 2 was that case too
  until the answer key was corrected (below): its rust is really in view.
  All five headline misses are that same undercut, reported in frames 1 to 4 as the
  corroded plate beside it, and the station 12 crack.
- Looking at frames 1 to 4, the undercut renders faintly: a thin line, not the dark groove
  the prompt describes. So the frames that test the undercut are weak tests of it.
- These are not independent trials. A frame shows 0.295 m of seam and the stations are
  0.1739 m apart, so 41 percent of each frame is also in the next, and the 15 sightings of
  a fault are 5 physical faults. The bounds treat each frame as a fresh draw, so they are,
  if anything, too kind.

**The probe.** Two of seven offsets at each of 24 stations, 48 measurements, against 28
rejectable points on 5 hidden patches. "Chance" is the exact probability that picking two
of seven at random finds at least as many. "Paired p" is an exact McNemar test against the
shipped sweep, one per run and never pooled.

| Planner | Rejectable found, of 28 | Hidden patches reached, of 5 | Chance | Paired p |
| --- | --- | --- | --- | --- |
| Oracle, knows the answers | 23 | | | |
| Sweep: coverage plus return to thin readings, no model, as shipped | **16** | 5 | 0.0002 | |
| the same sweep under all 5040 orders of its tie-break | 7 to 18, median 12 | | | |
| the same sweep, given only the state the model is shown | 11 (7 to 19 over tie-break orders, median 13) | 5 | 0.11 | |
| Nemotron 3 Nano, run 1 | 9 | 4 | 0.40 | 0.143 |
| Nemotron 3 Nano, run 2 | 12 | 4 | 0.047 | 0.344 |
| Nemotron 3 Nano, run 3 | 10 | 4 | 0.23 | 0.146 |
| Nemotron 3 Nano, camera hidden (one run) | 11; no change seen | 4 | 0.11 | 0.267 |
| Two of seven at random | 8 expected, central 95 percent 4 to 12 | | | |
| Always on the seam, then -0.15 | 2 | 1 | 0.9997 | |
| Probe beside reported rust | 2 | 1 | 0.9997 | |

The model was never ahead of the rule as shipped, in any run, and no single gap is
significant on its own (p 0.143 to 0.344); the 28 points cluster in 5 patches, so the pairs
are not independent either. Against the rule shown only the model's view it was not
always behind: run 2 found 12 to its 11, and the camera-hidden run tied at 11. Three of the
four model runs are results that random picks reach more often than one time in ten.
"Probe beside reported rust" is what steering the probe by the camera alone can do on this
rig: it finds 2, the same as never moving off the seam.

Run 3 includes one station, station 10, where the model's reply was unreadable and the rule
chose instead, so the model's own picks found 9, not 10. With station 10 left out of both
arms the paired p is still 0.146. Replies that named fewer offsets than the budget were
topped up from the rule, and these runs did not record where; live runs record it now.

`python tank/score_probes.py --tiebreaks` prints the tie-break distribution and
`python tank/layout_sweep.py` the layout sweep in the next section. Neither calls a model.

The inspector takes a median of 9.2 s a frame (mean 10.2 s) and the planner a median of
7.1 s a station, one model call each.

## How the headline changed, and why that is the result

The first version of this project claimed that an NVIDIA model planned the probing
better than arithmetic. It would have gone into the submission. It was wrong, and the
way it was caught is the most useful thing here.

The run originally held **four** rejectable probe points, from one hidden patch. On four
points Nano found three or four and the coverage rule found two, so the model appeared to
win. The exact paired tests on the three Nano runs give p 0.5, 1.0 and 0.5: not a near
miss, no evidence in either direction. On four points the smallest two-sided exact p
attainable is 0.125, one planner alone right on every point, so that design could not
have shown a difference at 0.05 whatever happened. `tank/significance.py` prints those
archived tests apart from the rest. Repeating it did not help, because every repeat
re-measured the same four points. Those runs are kept in
`tank/data/survey/archive_four_point/` as the record, and no result from them is used
here; Nemotron Super and a hybrid floor planner were measured only there, and appear
only in that archived table.

So the evaluation scene was given five hidden patches instead of one, taking it to **28**
rejectable points. Hidden loss adds no geometry, so the 24 scored frames stayed byte for
byte identical and every recorded verdict still applies. `tank/data/survey/frames.sha256`
records their digests, `tank/refresh_probes.py` recomputes the probe answers from the scene
without re-rendering and refuses to write unless every frame still matches, and
`git diff --stat 24208a0^ -- tank/data/survey/frames` prints nothing.

On 28 points the rule was ahead in every run: 16 against 9, 10 and 12. We first read that
as coverage beating reasoning. Two checks shrink the gap; one explanation we offered for it
did not survive.

The first check is the draw. The rule's 16 comes from one tie-break order: over all 5040
orders it finds 7 to 18, median 12, and 93.9 percent of orders find fewer than 16.

The explanation that failed is information. The model is shown the inspector's sentence
about the picture, which offsets exist, which have never been measured on the run, the
last six readings, and one nominal wall and reject level for the station. It is not told
how long ago each offset was last measured; the rule reads the whole run's history. Held
to the model's six readings and nothing more, the rule finds 11 of 28 at its shipped
order, and an earlier version of this file read that as the gap coming from the history
the rule reads and the model is not given. The tie-break study refutes it: over all 5040
orders the rule shown only the model's view finds 7 to 19, median 13, against a median of
12 with the full history.
Paired order by order, full history minus the model's view is median +0, mean -0.41; the
full history is ahead in 30.8 percent of orders, level in 19.7 and behind in 49.5.
Withholding the history does not hurt the rule, so the 11 is a draw like the 16, and
against that matched rule no run differs either (p 0.754 to 1.000).

The second check is the layout, which is favourable too. The hidden patches were placed
by hand after the four-point result, every one 0.27 to 0.41 m off the seam.
`tank/layout_sweep.py` puts the same five patches in 1000 random places (seed 1): the
sweep finds 57.1 percent of the rejectable points on the shipped layout against a median
of 40.7 percent, and 94.9 percent of layouts give it less.
The same layout punishes probing on the seam, 7.1 percent shipped against a median of 34.8.
On random layouts the sweep beats always-on-the-seam in only 572 of 1000.

Then the ablation. Replace the one sentence about the picture with "no camera on this run"
and the model finds 11 of 28, inside the range of its three sighted runs. That is largely
by construction: the camera images only the on-seam probe position, so on this rig there is
almost nothing for the picture to tell the planner. It is one run, so read it as no change
seen, not as a measured zero.

So nothing here shows a model choosing probe points better than a coverage rule, and only
weak evidence that it chooses them worse: across tie-break orders the rule typically
finds 12 to 13 against the model's 9 to 12, and no comparison is significant. That fits
published evaluations of language models as planners (PlanBench, Valmeekam et al.) and
the LLM-Modulo argument for pairing them with external checkers (Kambhampati et al.,
arXiv 2402.01817). The standard approach to this kind of sparse search is kriging, or
Gaussian-process thickness mapping, which measures next where the map is least certain.
We have not run it here, and it is the baseline a serious version of this comparison
should add.

The architecture this evidence supports is the one the README leads with: **the model
judges the picture, where there is something to see, and arithmetic drives the arm**,
because it is no worse, deterministic, free and instant. A model planner would also add
5.3 hours to a real tank, 4.0 even overlapped with driving, because each plan waits for the
last station's readings.

## How the answer key was corrected

The survey score this README used to lead with had no misses. It was scored against an
answer key that hid a crack visible in frame 12.

The old key projected every fault through the camera at the bead crown, 8 mm proud of the
plate. The crack is drawn lower, 2.4 mm proud, and shows on the plate above and below the
bead. Near the edge of the 14 mm lens that moved its projection to u = -0.006, just outside
a frame it is plainly in, at pixels 16 to 33 of 1280. The verdict at station 12 had been
recorded against "crack"; the old replay overwrote that with the key's "none" without
saying so.

The key is now rebuilt without Blender, in `tank/truth.py`. Seam faults are projected from
their rendered vertices (`tank/data/survey/fault_geometry.json`), and the camera model is
checked against Blender's own projection on 630 points
(`tank/data/survey/projection_check.json`, worst 0.02 px). `tank/refresh_probes.py`
rebuilds the key and refuses to write unless the frames match `frames.sha256`, and
`tests/test_tank_truth.py` pins all of it. The recorded verdicts were not touched, and
`--replay` prints any row where the answer stored with a verdict and the rebuilt key
disagree. Every inspection number above is scored against the corrected key.

Then the key turned out to be wrong a second way. It measured a fault's share of the
picture on a grid of wall points round the crawler's origin, but the lens sits 0.46 m ahead
of the origin, so that grid covered only the left sixth of every frame. `tank/truth.py` now
samples the picture itself, 64 by 36 rays cast onto the wall and tested against the
rendered outline of every visible rust patch, and a test checks that share against the
rust-coloured pixels of every frame. Frame 2's rust is really in view, 0.16 of the picture
by the key and 0.12 by pixels, not a sliver, so the strict row rose from 21 to 22 of 24. No
worst-fault answer moved, and `--replay` prints station 2 as the one frame where the key now
differs from the key its verdict was recorded against.

This is the third time the project has corrected its own scoring: the four-point probe
headline, the station 12 crack, and now the share of frame 2.

## What is still wrong with it

- **Station 12.** The inspector passed a frame holding a 64 mm crack, the same crack it
  named in frame 11. The run still flagged the crack, but one frame's pass is not a clean
  bill for the seam it shows.
- **Undercut.** Four frames carry D001, a 1.7 mm groove at the weld toe against a 0.8 mm
  limit. The inspector reported the corroded plate beside it every time and never named
  the groove. In those frames the groove renders faintly, a thin line rather than the dark
  groove the prompt describes, which may be why; the render and the model are not
  separated, and the recorded verdicts are one model's. Written up, not tuned away.
- **A development set.** The scene was corrected after reading the model's answers (commit
  048b919, six faults fixed). There is no held-out survey, so every inspection number here
  is a development-set number.
- **Choices that favour the inspector.** The four weld faults share one near-black
  material, and the prompt describes the crack, the undercut and the porosity as dark.
  The undercut is the exception in the pictures: it renders faintly, so for that fault the
  choice did not favour the inspector. "Declined" is only allowed when no bead is in the
  frame, so "none declined" holds by construction. The counting threshold matters at the
  edges: re-scored with it moved, the walk-in count stays 23 for anything from 0.01 to 0.2
  of a fault or of the frame (the scorer uses 0.08), and is 22 at 0 and at 0.25 and above.
  `tank/significance.py` prints the whole sweep.
- **Overlap and clustering.** Frames share 41 percent of their seam with the next, so 24
  frames are not 24 trials, and the 28 probe points sit in 5 patches, so neither are they.
- **The bead is exaggerated for visibility**, about 2.8 times the API 650 reinforcement
  limit for this joint, so its toe shadows read at render resolution.
- **The probe cross-slide is notional.** The scoring assumes the probe can be put down at
  seven offsets up to 0.45 m either side of the seam; nothing models that mechanism or its
  time, and the two-in-seven budget is an assumption.
- **Adhesion is an envelope, not a margin.** `tank/adhesion.py`, at 6.80 kg with 6 m of
  tether and an assumed 220 N rated pull a wheel: dry (friction 0.35) it holds safety
  factor 3 at 0.2 mm liftoff on every course, and fails it at 1.2 mm everywhere and at
  0.6 mm on 6 mm plate. With couplant on the wheels (friction 0.2, assumed) it is below
  safety factor 3 in every case and slides at 1.2 mm on 6 mm plate. Holding safety factor
  3 in the worst dry case needs about 435 N a wheel. The wheels have to be kept out of the
  couplant or rated higher.
- **One inspection pass.** Each frame was judged once, so the inspector's spread across
  repeats is not measured. The planner ran three times sighted and once blind.
- **The model is the bottleneck, not the robot.** A real 473 m shell is 2720 stations:
  1.3 hours of driving at an assumed 0.10 m/s and 7.0 hours of serial vision inference at
  the median of 9.2 s a frame, 7.7 hours at the mean of 10.2 s. About six requests in
  flight would keep the inference at drive speed, at either figure, and
  nothing here does that yet. A model planner would add 5.3 hours (4.0 overlapped), one
  more reason arithmetic drives the arm. Cost per tank is not yet measured.
- **The world is ours.** We built the tank, planted the faults and wrote the answers. That
  is how a simulated test works, but it means every number here is a statement about this
  scene, and the evaluation scene deliberately carries more hidden loss than a real tank
  of this age would.
- **Small samples.** 24 frames and 28 rejectable points. Enough to rule some things out;
  not enough to rank closely matched planners, and no single probe comparison here is
  significant on its own.
- **No hardware yet.** A bench plate with machined back-face pockets and a real gauge is
  specified in `docs/bench/`. Nothing physical has been measured.

## The same discipline, first measured on a welding cell

The loop was built and measured on a robotic welding cell before it was pointed at a tank
wall. It is here as evidence that the method transfers, and because it is where most of
the measured numbers in this repository come from.

![Weld seam found to under a millimetre on parts arriving centimetres out](docs/figures/seam_result.png)

Every row is one part. The pale dot is where a taught program would have welded; the
dark dot is where the cell put the seam after measuring it. The arrival scatter is
exaggerated for visibility; on a real line the joint moves with part variation such as
camber, springback and fit-up. The 0.67 mm is the tracker on exact rendered depth; with
calibration and the rest of the chain the cell's error budget (`cell/error_budget.py`)
puts the weld near 2.8 mm with a typical stereo camera and 1.2 mm with a laser profiler,
against a 1.5 mm requirement, which this simulation cannot yet measure. Drawn from the
campaign file by `scripts/draw_result_chart.py`, so a number on the chart cannot drift
away from the run that produced it.

## Run it

```bash
pip install -r requirements.txt
uvicorn service.app:app            # leave running; open http://127.0.0.1:8000 (the tank page)
```

In a second terminal:

```bash
python scripts/demo.py --offline   # three seconds, and the cell is running
```

For the welding cell: no GPU, no API key and no download. Six parts of captured depth
ship with the repository, and the demo runs the actual pipeline over them rather than
replaying saved numbers. Without `--offline`, a Token Factory key turns on the half that
calls a model.

The tank run. None of these calls a model or writes a file, with a key or without one:

```bash
python tank/score_survey.py --replay      # re-mark the recorded verdicts
python tank/score_probes.py --baselines   # the arithmetic planners, chance and the oracle
python tank/score_probes.py --tiebreaks   # the sweep under every order of its tie-break
python tank/layout_sweep.py               # the same planners on 1000 random patch layouts
python tank/significance.py               # bounds and the paired tests
```

Asking a model again is a separate flag, `--live`, on `score_survey.py` and
`score_probes.py`. It is paid, one Token Factory request a frame or a station, and it
writes into `tank/data/survey/runs/` rather than over the recorded files.
`score_probes.py --live --blind` hides the camera from the planner.

All of them read the captured dataset in `tank/data/survey`, which ships with the
repository. Rebuilding the tank itself needs Blender running with its socket add-on on
port 9876:

```bash
python scripts/blender_send.py tank/sim/blender_tank.py   # 473 m of seam, six courses
python scripts/blender_send.py tank/sim/capture_tank.py   # 24 stations, pictures, depth
```

![The dashboard's tank page](docs/figures/dashboard.png)

## What was measured

| | Result | Scored against |
| --- | --- | --- |
| Seam tracked along its whole length | **22.8 mm → 0.67 mm**, 50 of 50 parts under 1 mm, on exact rendered depth | the true seam, 1,000 stations |
| Choosing the model size for the per-part decision | 88 of 89 for the smallest model; the one miss accepted a view a second look would have improved by 0.3 mm. The same limits as arithmetic: 86 of 89, exact paired p 0.625 | ground truth, not a larger model's opinion |
| Post-weld inspection | 91% pass or defect on 150 beads, 111 of 150 named exactly, no false alarms | the defect that was injected, before the model saw the picture |
| Thermal compensation | compensating beats not compensating from -40% to +100% model error, 0.22 against 0.33 mm and 0.56 against 1.10 mm at the ends; 0.03 mm only when the model is exactly right | a distortion model calibrated to published figures, of the same form the compensator uses |
| Which machine is drifting, on easy cases | 16 of 16 | **retired, see below** |
| Which machine is drifting, on hard cases | best model 72%, the cell's default 70%, a fixed rule 56% | cases sampled onto the decision boundary |

### If you read three things

**[The test the answer key passes](#the-test-the-answer-key-passes-and-what-replaces-it).**
Every drift score here was measured on cases where the answer is recoverable by
comparing two numbers to a threshold, and four lines of arithmetic score full marks on
them in zero milliseconds. The replacement is a set built by rejection sampling onto
the boundary. Six hosted models were run on it. The best gets 72% against the rule's
56%, one of them scores below the rule, and the fine-tuned model's perfect score turns
out to be an artefact of the easy set. The test itself then turned out to have a hole
in it, which is written up with the rest.

![Nine arms against five families of hard case](docs/figures/boundary_test.png)

**[A drifting camera looks exactly like a drifting fixture](#a-drifting-camera-looks-exactly-like-a-drifting-fixture).**
Measured on the seam alone the two faults are not similar, they are the same numbers,
to two decimal places. No model fixes that because the information is not in the data.
One reference feature bolted to the machine base does: same model, same prompt, 0 of 4
drifting cameras identified becomes 4 of 4.

**[What the 0.67 mm does not include](#what-the-067-mm-does-not-include).** The tracker
finds the joint to 0.67 mm. Add hand-eye calibration, robot repeatability, standoff
drift and what the arc itself does, and the chain comes to 2.82 mm against a 1.5 mm
requirement. The largest term is calibration, it is systematic, and this cell cannot
measure it.

Every number in these tables was measured on the simulated cell. Where a claim turned
out to be wrong, it has been retracted on the spot rather than quietly deleted, with the
test that disproved it printed next to it.

## The problem

A welding robot is taught a path once and then drives that same path on every part.
That works if every part arrives in exactly the same place. They do not.

These parts start as flat sheet metal. The sides are folded up in a press, and every
fold springs back by a slightly different amount, so no two parts come out identical.
The joint that needs welding, which is the line where two folded edges meet, ends up
somewhere different on each one. In the simulation the parts also arrive scattered, up to
±30 mm across and ±2 degrees in yaw. That scatter is exaggerated for visibility, so the
correction reads on a chart, and it is most of the taught path's miss here: across the
fifty parts measured, 23 mm on average and 51 mm at worst. The part variation sits on top
of it, and it is the part that matters on a real line: a locator can take out where a part
sits, but not how the press bent it. The weld would land on flat panel instead of in the
joint.

There is a second problem on top of that. The joint is not just in the wrong place, it
is also slightly curved: the press leaves it bowed. So even a robot that found the start
and end of the joint perfectly, and drove a straight line between them, would still
wander off it in the middle.

The usual fixes are to hold tighter tolerances at the press, or to have a skilled
operator watch the line and correct it. Both cost money, neither scales, and neither of
them straightens a joint the press has already bent.

This project does something else: it assumes the part will be in the wrong place, and
measures where it actually is before welding.

![The cell in plan and elevation](docs/figures/cell_layout.png)

The cell, drawn to scale from the same constants the simulation builds from by
`scripts/draw_cell.py`. No renderer is involved: this is the model, not a picture of
it. The camera stands 0.95 m off the joint, the joint runs 450 mm top to bottom, and
the panels are 2 mm sheet.

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

![One part, tracked from its depth frame](docs/figures/seam_tracked.png)

One part from the shipped capture, drawn by `scripts/track_on_model.py` against
`data/demo/manifest.json`, so this picture regenerates on a clone. Grey is the taught
path, red is where the seam really is, and the dots are the measurement at each of the
twenty stations, coloured by whether the edges are touching there or the joint has
opened. The taught path misses this part by 34.2 mm. Everything in the image comes out
of the depth buffer, including the shading: no colour pass is involved at any stage.

**2. Decide whether to believe it, in the cloud.** One look is not evidence. The
measurement comes with signals a real cell has: the angle between the fitted faces, how
evenly the two faces were seen, the fit residual on each, how much of the joint was
covered. NVIDIA Nemotron 3 Nano reads those and decides whether the measurement can
be trusted or whether the part should be presented differently and measured again.

What this catches, measured: across fifty parts the worst individual views came back
seventy millimetres out, and every one of them was the part turned forty five degrees,
where the second face goes nearly edge-on and the fitter locks onto a locating pad
instead. Those views do not hide it. They arrive carrying an inlier balance of 0.10 and
a forty five degree included angle where a square corner should read ninety, and the
model reads exactly those signals. Of eighty nine views taken, sixteen were badly wrong
or refused outright, and all sixteen were kept out of the fused answer.

**3. Send each decision to the model that is actually right about it.** Four Nemotron
sizes are served on Token Factory, and all four were scored on the per-part decision
against ground truth. The smallest gets 88 of 89 and is the one that runs; nothing is
escalated to a larger size, because none of them did better and the 120B did worse. The
router still records what the evidence looked like, for the operator. That is the
opposite of the obvious design, and the numbers are under Validation. The limits the
prompt states, written as arithmetic, get 86 of 89 on the same points, so this test does
not show Nano beating a rule either.

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
bead and a vision model, DeepSeek V4.1 Flash, returns a verdict with its reasoning:
pass, or a named defect. Defects are deliberately injected in simulation, so every
verdict is scored against a known answer rather than admired.

**7. Supervise, and diagnose which machine to go and touch.** Nemotron 3.5 Lightning runs
as a cell supervisor with tools over the production database, and the once-a-shift
diagnosis is asked of DeepSeek V4.1 Flash, one case behind the best arm on the boundary
test below and already the cell's bead inspector, so the diagnosis adds no model the cell
does not already call. Its margin over a fixed rule does not survive correction for the
nine arms tested, so the choice does not rest on significance. Spotting that something is
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

Both of those were, for a long time, the parts of the cell nobody had checked. They
ran, and whether what they said matched what their own tools returned was taken on
trust. They are scored now, against the database they answer from.

| | Result | Before the two fixes below |
| --- | --- | --- |
| Supervisor, on ten questions the database answers exactly | 9 right, 1 wrong | |
| Briefs: came back at all | 6 of 6 | 5 of 6 |
| Briefs: carried the measured figure it was handed | 6 of 6 | 5 of 6 |
| Briefs: said what to do | 6 of 6 | 5 of 6 |
| Briefs: cited a source it was actually given | 6 of 6 | 3 of 6 |

The one supervisor answer marked wrong is the open question, "should we hold the line
right now", where it summarised the state of the cell accurately and never committed to
an answer. Being unwilling to say yes or no is a real failure for a supervisor, so it is
scored as one rather than excused.

The brief writer first scored three in six on citation, which was the worst number in
this repository. Reading the three it failed, all three cite properly, with numbered
markers and a source list, and what they were being punished for was **converting
units**. The model writes "if root opening exceeds 1/16 in (1.6 mm)", the cited source
contains the string 1/16 and never the string 1.6, and a check comparing figures as text
called the conversion an invented number. It is the opposite of invented. It is the
model doing the arithmetic a welder would otherwise do at the bench, for a shop that
works in millimetres. The check now reads fractions and accepts a correct conversion.

The sixth brief never came back at all, with a JSON parse error four lines in. The same
inch mark did it: a model writing 1/16" inside a JSON string closes the string. The
prompt now asks for "in" and forbids the prime mark, and says why.

Both faults were in the harness, which is where almost every fault in this project has
turned out to be. Run it yourself with `python scripts/eval_briefs.py --limit 6`.

The division is deliberate. Geometry does what geometry is good at, which is turning
surfaces into millimetres. The models do what geometry cannot: judge whether a
measurement deserves trust, judge whether a weld is sound, diagnose a machine from a
run of parts, and explain any of it to the person responsible for the line.

## How Nebius and NVIDIA are used

| Layer | Platform | Model | Runs today |
| --- | --- | --- | --- |
| Where to put the thickness probe next, in the tank | Nebius Token Factory | Nemotron 3 Nano 30B-A3B | measured against the sweep rule; the rule is used |
| Reading a tank seam from the crawler's camera | Nebius Token Factory | DeepSeek V4.1 Flash | yes |
| View planning, inside the control loop | Nebius Token Factory | Nemotron 3 Nano 30B-A3B | yes |
| Shift-level diagnosis, which machine slipped | Nebius Token Factory | DeepSeek V4.1 Flash | yes |
| Cell supervisor agent, with tool calling | Nebius Token Factory | Nemotron 3.5 Lightning | yes |
| Post-weld inspection | Nebius Token Factory | DeepSeek V4.1 Flash | yes |
| Writing a corrective action brief from fetched documents | Nebius Token Factory | Nemotron 3.5 Lightning | yes |
| Shift-level diagnosis, without a network | trained on Token Factory, runs on the cell's GPU | Qwen3-1.7B + LoRA | scored; weights to be published with a release |
| Looking up standards the cell cannot measure | Tavily | search | yes |
| Reading the documents a fault raises | Tavily | search, then extract | yes |

Every hosted model call runs on Nebius Token Factory, through one shared client in
`brain/nebius.py`; the offline diagnosis model was trained there and runs on the cell.
The models are named once, with their roles, in `brain/registry.py`, and the defaults are
set in `brain/nebius.py`. Four NVIDIA Nemotron sizes are served on Token Factory. This
project runs two of them in the loop, Nano for every per-part decision and Lightning for
the supervisor and the briefs, and measured all four, because measuring them showed they
are not interchangeable. Super and Ultra were scored and are not called in the loop. A
row that says "yes" runs today; a row that does not, says what it is instead.

The fine-tuned edge model's adapter is about 81 MB and is kept out of git, so until it is
attached to a release its scores are the evidence and the weights cannot be rebuilt from a
clone; `scripts/eval_drift_edge.py` says so when the adapter is missing.

The tank's probe planner calls Nano with default reasoning. The view planner and the brief
writer pass `reasoning_effort="none"`, which `brain/viewplanner.py`
records as the setting that makes Nemotron answer rather than reason in prose. So the
recorded planner runs were made in a different envelope from the cell's own Nemotron
calls, and they are reported as they were made. The envelope a call is made in varies by
model. What is asked does not.

The inspection model is not an NVIDIA one, and that is a constraint rather than a
preference. Every model the account serves was sent one clean bead and one with a hole
burnt through it, along the real inspection path (`python scripts/probe_vision_field.py`,
result in `data/vision_field.jsonl`):

<!-- vision-field:begin -->
| What happened | Models |
| --- | --- |
| refuses any picture (15 of 23) | deepseek-ai/DeepSeek-V4-Pro, deepseek-ai/DeepSeek-V4-Pro-0813, nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B, nvidia/nemotron-3-super-120b-a12b, nvidia/Nemotron-3-Ultra-550b-a55b, nvidia/Nemotron-3_5-Lightning, NousResearch/Hermes-4-405B, Qwen/Qwen3-235B-A22B-Instruct-2507, Qwen/Qwen3-30B-A3B-Instruct-2507, Qwen/Qwen3.5-397B-A17B, deepseek-ai/DeepSeek-V4-Flash-0731, openai/gpt-oss-120b, zai-org/GLM-5.1, zai-org/GLM-5.2, zai-org/GLM-5.3 |
| takes it, will not judge a rendering (2 of 23) | MiniMaxAI/MiniMax-M3, moonshotai/Kimi-K3 |
| judges it (6 of 23) | deepseek-ai/DeepSeek-V4.1-Flash, zai-org/GLM-5.3-Flash, google/gemma-3-27b-it, moonshotai/Kimi-K2.6, openbmb/MiniCPM-V-4_5, moonshotai/Kimi-K2.7-Code |
<!-- vision-field:end -->

Of the 23 models the key reaches, 15 refuse the image outright, all four Nemotron sizes
among them; 8 accept it, and 6 of those judge the bead. `service/evidence.py`
(`vision_field_study()`) counts those from `data/vision_field.jsonl`. Reasoning about the per-part
measurement, which is text, therefore runs on Nemotron; reasoning about the picture runs
on the best of the six that will look at it.

"Best" is measured, and the first measurement was wrong about the models. The prompt as
first written called the picture a photograph and offered "uncertain" as a way out. The
bead pictures are renders from the simulator, and the stronger models said so and took
the way out; Gemma and MiniCPM played along. That was the prompt's fault, not the
models'. So the same fifty beads were run again under a prompt that tells every model
what it is looking at, and once more with the bead shown up close, which is the window
a bead camera would be aimed at. Same pictures, same question, same scoring; only the
framing changed, and it changed for every model. Each cell is defects named right, of
the beads the model would judge:

<!-- vision-variants:begin -->
| | prompt as first written | told it is a rendering | told it is a rendering, shown the bead up close | and a hole defined apart from a pore | and the hole drawn as a hole, on 150 beads |
| --- | --- | --- | --- | --- | --- |
| **DeepSeek V4.1 Flash** | 17 of 26 judged, 24 declined; 1 missed | 35 of 50 judged; 3 missed | 35 of 50 judged; 3 missed | 36 of 50 judged; 3 missed | 111 of 150 judged; 14 missed |
| Gemma 3 27B | 31 of 50 judged; 4 missed | 35 of 50 judged; 3 missed | 36 of 50 judged; 4 missed | 36 of 50 judged; 4 missed | 106 of 150 judged; 18 missed |
| MiniCPM-V 4.5 | 31 of 50 judged; 19 missed | 31 of 50 judged; 19 missed | 31 of 50 judged; 14 missed | 31 of 50 judged; 19 missed | not run |
| Kimi K2.6 | 31 of 40 judged, 10 declined; 7 missed | 32 of 46 judged, 4 declined; 9 missed | 33 of 42 judged, 8 declined; 5 missed | 38 of 45 judged, 5 declined; 4 missed | not run |
| GLM-5.3 Flash | 12 of 21 judged, 29 declined; 2 missed | 35 of 49 judged, 1 declined; 3 missed | 33 of 44 judged, 6 declined; 3 missed | 36 of 46 judged, 4 declined; 4 missed | not run |
| MiniMax-M3 | 1 of 1 judged, 49 declined; 0 missed | 34 of 50 judged; 9 missed | 37 of 50 judged; 4 missed | 39 of 50 judged; 6 missed | 117 of 150 judged; 20 missed |
<!-- vision-variants:end -->

Told the truth, nobody declines and every model improves. One more fault in the
question was found the same way, by reading the wrong answers: every model called a
burn-through "porosity". The simulator draws burn-through as one dark hole in an
unbroken bead, and the definitions said porosity was "small dark rounded cavities", so
the models were reading the picture correctly and being marked wrong for it. The
definitions now separate one large hole where the bead is missing from several small
pits, and the renderer now stops the bead either side of the hole. The fifty parts were
re-rendered with the corrected hole and a hundred more rendered at a higher defect rate,
150 beads with 91 defects, enough to tell apart three models that tied on fifty. The
three were scored on them under the prompt the cell now uses:

<!-- vision-beads:begin -->
| | Beads judged | Named right | Pass/defect right | False alarms | Missed | Declined | Defect types named | Median time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **DeepSeek V4.1 Flash** | 150 of 150 | 111 | 136 | 0 | 14 | 0 | 4 | 3.2 s |
| MiniMax-M3 | 150 of 150 | 117 | 130 | 0 | 20 | 0 | 5 | 2.8 s |
| Gemma 3 27B | 150 of 150 | 106 | 131 | 1 | 18 | 0 | 5 | 1.9 s |
<!-- vision-beads:end -->

On fifty beads the three tied. On 150 they do not. On the pass-or-defect call, which
is what the cell routes a part on, DeepSeek V4.1 Flash is right on six beads where
MiniMax-M3 is wrong and never the reverse, a paired p of 0.031, and it ships 14 bad
parts to MiniMax's 20 with no false alarms from either. MiniMax still names the defect
best, 117 to 111, which is advice for the person who fixes it, not the gate. The cell
inspects with DeepSeek V4.1 Flash, at 3.2 seconds a bead. The burn-through fix held:
23 of 24 named right where every model had called it porosity.

The larger set also showed the next render fault, the same way the first two were
found, by reading the wrong answers. Every model calls a skip clean, 0 of 13, and calls
an undercut a skip, because the heat-tint bands were drawn the full length of the joint
whatever the bead did, so a gap in the bead never showed, and the undercut groove sat
on the corner itself where it read as a break in the bead. The renderer now draws the
tint only where a bead was laid, shows the bare joint in a gap, and puts the groove
beside the bead on the plate; the beads scored here are the renders before that change,
and the numbers for skip and undercut are the render's failure, not the models'.

One thing the comparison had to be fair about: a reasoning model spends its output
budget thinking before it answers, and under the 400 tokens the inspector sends it came
back with nothing, which the parser scored as a wrong verdict. So a reply that did not
parse and spent its budget reasoning is sent once more with room. The prompt, the
pictures and the scoring never change per model; only the envelope does.

Cosmos is not served on Token Factory at all. Reaching Cosmos Reason or Cosmos Transfer
means self-hosting on Nebius AI Cloud, which is a different platform and a larger piece
of work. It is the obvious next step and no result from it is claimed here.

### One model size does not fit one cell

The obvious architecture escalates: a cheap model on routine calls, a large one when
the evidence conflicts. That is what was built first, and measuring it took it apart;
nothing escalates now, because the larger sizes were measured to be no better at this
question and the 120B is worse.

Scored against ground truth rather than against the largest model's opinion, on 89
decision points where the simulation knows the answer, under the prompt the cell runs:

| Model | Correct | Looks it did not need | Failed views accepted | Model time for the run |
| --- | --- | --- | --- | --- |
| Nemotron 3 Nano 30B | **88 of 89** | 0 | 1 | 132 s |
| Nemotron 3.5 Lightning | 88 of 89 | 0 | 1 | 98 s |
| Nemotron 3 Super 120B | 61 of 89 | 28 | 0 | 150 s |
| Nemotron 3 Ultra 550B | 88 of 89 | 1 | 0 | 141 s |

The decision set has both answers in it, which the first version of this test did not.
Thirty-nine points are replayed from the campaign, where every part was seen square-on
first and a second look was never worth taking. Fifty more take a captured part and
show it first from one of its five angles, twenty-five where the first view failed to
measure or landed tens of millimetres out and a look is needed, twenty-five where it
did not. A model can now be wrong two ways: ask for a look it does not need, which costs
a robot move, or accept a view that a look would have fixed, which welds the part on it.
Three of the four sizes get all but one right; the one miss for the two smaller models
is a view 22.5 degrees off square that a look would have improved by a third of a
millimetre, the boundary case. The smallest one runs because it is the cheapest of the
three. The 120B still asks for a second look on a third of the parts with the boundary
stated, and that is the one model-level finding in the table.

The prompt states its limits as numbers, so the same limits can be applied as arithmetic
(`rule_decision` in `brain/viewplanner.py`) to the same 89 decision points, with no model
call:

| | Points | Rule right | Nano right | Rule alone right | Nano alone right | Exact McNemar p |
| --- | --- | --- | --- | --- | --- | --- |
| Replayed from the campaign | 39 | 39 | 39 | 0 | 0 | 1.0 |
| First view from one of five angles | 50 | 47 | 49 | 1 | 3 | 0.625 |
| All | 89 | 86 | 88 | 1 | 3 | 0.625 |

The rule is right alone at the point Nano misses, where one face fits at 0.584 mm against
a 0.5 mm limit. Nano is right alone at three views whose inlier balance is 0.508 to 0.586
against a 0.60 limit, where a second look would have gained at most 0.11 mm and at two of
them made the answer worse; its stated reason at all three says the balance is at least
0.60, so it was right there while misreading the number the rule reads. Four
disagreements on 89 points cannot separate the two. So Nano makes every per-part
decision because it was measured at least as good as the rule on this set, not because
it was measured better.
`python scripts/eval_router_truth.py --rule` pairs the rule with the recorded answers in
`data/router_truth_eval.jsonl`, with no model call. The 39 replayed points rebuild exactly
from the campaign. The 50 first-view points were measured on a render that is no longer on
disk, so their signals are rebuilt from a re-render of the same parts and kept in
`data/router_rule_eval.json`; where the re-render is present the script rebuilds them and
rewrites that file. It scores only if the answer key agrees on all 50 points and none of
the 363 readings the models quoted in their reasons, put in one at a time, moves the rule's
verdict. The kept signals are not the ones the models saw: 3 distinct quoted readings
differ from them by more than 5 percent.

That is not the first table this section carried, and the reason it changed is worth
keeping. The prompt as first written said:

> Accept a single view only if its signals are excellent and no other view is available.

Every part in the campaign starts with exactly one view, and "excellent" is never
defined anywhere in the prompt. Read literally, that sentence instructs a model to ask
for a second look on every part in the set, which is precisely what Super did, 39 times
out of 39. Scoring that as a judgment failure was scoring obedience to an instruction
with no number in it. Under that prompt the 39 replayed decisions scored Nano 34,
Lightning 1, Super 0 and Ultra 21 of 39, and this file called the smallest model the
most correct one and told a story about large models being cautious.

Putting the same decisions to the same models with one sentence added, giving the
boundary a number taken from what the cell already treats as a good view rather than
tuned against the answers:

| Model | Prompt as written | Boundary stated |
| --- | --- | --- |
| Nemotron 3 Super 120B | 0 of 39 | **20 of 39** |
| Nemotron 3.5 Lightning | 1 of 39 | **39 of 39** |
| Nemotron 3 Nano 30B | 33 of 39 | **39 of 39** |

The left column reproduces the routing study to the decision: 0 of 39 and 1 of 39 are
exactly what that table reports, because this runs the same cases through the same
planner and scores them the same way. Only the sentence under test changes. The first
version of this script did not do that. It rebuilt the cases and the call by hand, and
returned 18 of 20 for Lightning against the study's 1 of 39, which was a difference
between two harnesses rather than between two prompts. A number that disagrees with the
thing it is meant to explain is not a finding, it is a bug, and it was one.

So the number went into the prompt the cell runs, and the table at the top of this
section is measured under it. The small model is still the right one to run, now for the
plain reason that it is the cheapest of the three that score 88 of 89, and a
reply the planner cannot read falls back to arithmetic rather than to accepting the
measurement. Until this audit that fallback looked for the signals under different keys
from the ones the cell passes, so it asked for another look on all 89 points. It erred
toward a robot move rather than toward welding on a bad view; it is now fixed and pinned
by `tests/test_router_rule.py`. But "large models are inherently
more cautious" was a story told about a prompt defect. Every model in the family
improves once the boundary has a number, and two of them go to full marks, which is not
what a difference in caution looks like. A reader entitled to the first table is entitled to this one.

Two properties of the cell also make the small model's answer safe rather than lucky:
accepted views are fused with a median, so one bad view is outvoted before any model is
consulted, and on the 39 parts replayed from the campaign, all seen square-on first, there
was nothing for a second look to find.

This is the second time this evaluation has been wrong about why. The first version
compared each tier against the largest model and reported that the small one agreed with
it zero percent of the time, which measured conformity rather than quality. Both errors
had the same shape: a number was produced, and the explanation attached to it was never
itself tested.

```bash
python scripts/eval_view_prompt.py --cases 20
```

Where the sizes DO separate is the other job. Asked to read thirty parts of measurements
and say whether a press or a fixture is drifting, the small models answer "press" every
time regardless of the data, and do not improve when the signal is made clearer. The
large ones read it:

| Model | Marginal drift | Clearer drift |
| --- | --- | --- |
| Nano, Lightning | 4 of 12, answered "press" 12 of 12 | 4 of 12, unchanged |
| Super, Ultra | 8 of 12 | 10 of 12 |

So the split that survives is by task, not by evidence difficulty. A fast judgement
over a handful of numbers goes to the smallest model at about 1.4 seconds. Trend
reasoning across a run of parts goes to a model that reads the trend: on this
three-cause question that meant a large Nemotron, and on the harder boundary test below
the cell's default became DeepSeek V4.1 Flash. Sending the second job to the small
Nemotron produces a confident answer that is the same every time.

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

Fifty parts arrive under physics with a randomised arrival offset, up to ±30 mm and
±2 degrees, exaggerated for visibility, and the press leaves residual springback and
camber on every folded side, which is the part variation a real line has. Every arm reads
the same renders of the same parts.

### Following the seam

The measurement that matters: not where the corner is as a point, but where the seam
runs along its length, taken at twenty stations up each part. A thousand measurements
over fifty parts.

| | Mean | p95 | Worst | Sub-millimetre |
| --- | --- | --- | --- | --- |
| Taught path, no correction | 22.80 mm | 36.37 | 50.91 | 0 of 50 |
| Tracked from depth | **0.67 mm** | 0.83 | **0.89** | **50 of 50** |

That is the tracker, on exact rendered depth. It is one link in a chain, and the
whole chain comes to 2.8 mm on a realistic sensor. See "What the 0.67 mm does not
include" below before quoting it as the accuracy of the cell.

Every part tracked, every station, none refused. The worst single station across all
thousand reads 1.05 mm.

These figures come from the campaign file, `data/seam_campaign.jsonl`, which was run with
the slice tracker, before each face was fitted once from all its points (see "Where it
stops working" below). The same clean campaign with the current fit gives 0.676 mm and a
worst station of 1.03 mm. The demo runs the current tracker, so its per-part figures do
not match the campaign file exactly.

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
| Loop, every decision to Nano (routed) | 0.91 mm | 1.61 | 1.79 | 26 of 50 |

Routing costs 1.00 looks per part against 1.78, which is
44 percent fewer robot moves, and 74 seconds of
model time against 206, for 0.021 mm of accuracy. The routed file,
`data/campaign50_routed.jsonl`, records Nemotron 3 Nano on all 50 of its decisions. The
fixed-model file, `data/campaign50.jsonl`, does not record which model its arm called.

The honest reading on the loop itself is that it is not currently earning its cost on
this measure. An earlier run showed a large gap, and that gap closed when three faults
were fixed in the capture: a camera near plane clipping the corner out of every depth
image, an end stop standing on the seam, and a sheet thickness no press could fold.
Those fixes made the single fixed view good. What the loop demonstrably still does is
refuse bad evidence, and what routing does is send every decision to the smallest model,
at 1.00 looks a part against 1.78.

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

The short version of the cloud result: this task needs a large model, and the small
ones answer "press" every time, which scores 4 of 12 without reading anything.

That was the finding. It was also a problem, because it put the one question a shop
actually pays for behind a 120B model and a network connection.

**So the model was taught instead of bought.** The cell is a simulation with a forming
model in it, so a run of parts with a known fault costs nothing to make and arrives
already labelled. Six hundred runs were generated, a 1.7B model was fine-tuned on
Nebius Token Factory in four minutes, and the adapter now runs on the workstation beside
the cell with no network at all.

| Model | Where | Result | How it got there |
| --- | --- | --- | --- |
| Qwen3-1.7B, untrained | on the cell | 4 of 12 | answered "none" every time |
| Nemotron 3 Nano 30B-A3B | Token Factory | 4 of 12 | answered "press" every time |
| Nemotron 3.5 Lightning | Token Factory | 4 of 12 | answered "press" every time |
| Nemotron 3 Super 120B-A12B | Token Factory | 8 of 12 | read the evidence |
| Nemotron 3 Ultra 550B | Token Factory | 8 of 12 | read the evidence |
| Qwen3-1.7B + LoRA | on the cell | 12 of 12, on the easy three-cause set | trained on simulated runs |

The 12 of 12 is on the easy kind of set that the boundary test below was built to
replace, so read it as the training working on this question, not as diagnosis.

The comparison is kept honest by construction. The same generator (`cell/drift.py`)
builds the training runs and the scored runs; the four seeds the evaluation lands on are
excluded from the training set with a margin either side; the system prompt is imported
rather than retyped, so the trained model is not given a better prompt than its
baselines; and reasoning is disabled for it exactly as it was for the Nemotron sizes.
The untrained control on the same base model and the same prompt is the line that says
the training did the work.

One caveat worth stating: the trained model answers in about 4 s on a laptop GPU against
roughly 2.3 s for the 120B in the cloud. It is slower per call and available when the
line is not, and it is asked once a shift, so the wall clock barely matters.

### Where it stops working

Every number above was measured on exact rendered depth, which makes them ceilings
rather than results. A joint finder scored on perfect input has not been scored on
anything, so the whole fifty-part campaign was run again at twenty-eight sensor
degradations, twice over, and the error followed until it broke. The threshold was
fixed before the sweep ran: the wire is 1.0 mm across, so a path further than that
from the joint is not on the joint.

The sensor model is not "add some Gaussian noise". It is the three things that
decide this particular measurement: clustered specular dropout, because folded
sheet steel returns nothing from a face angled away and does it in patches rather
than speckle; flying pixels at depth discontinuities, which matter here more than
anywhere because the discontinuity IS the joint; and range noise scaling with the
square of distance, because disparity resolution is fixed in pixels.

The result inverts the obvious intuition:

| Failure mode | Holds to | Fails at |
| --- | --- | --- |
| Specular dropout | **90% of the frame gone** | never, across the sweep |
| Flying pixels at the joint edge | **90%** | never, across the sweep |
| Range accuracy at one metre | 1.0 mm | **2.0 mm** |

The killer is accuracy, not missing data. The method shrugs off losing most of the
frame and dies on two millimetres of range error.

Chasing that turned up a bug and produced the one algorithmic change in this
repository. The bug: the RANSAC inlier band for the face fits defaulted wider than
the sheet is thick, so one plane could claim a panel's inner AND outer surface and
put its offset between them, biasing the seam by about 1.4 mm. The change: a
station used to fit its two lines from the few hundred points inside a thin slice,
and a short segment constrains a line's ANGLE badly, which is then multiplied out
to the vertex. But the faces are two continuous panels seen in tens of thousands of
points, and a folded panel has one orientation. Only how far out it sits varies,
because springback bows it. So each face is now fitted once from every point it
has, with only its offset free as a low-order polynomial in height.

What that bought, worst station on any part in brackets:

| | Slices alone | Faces fitted once |
| --- | --- | --- |
| Clean | 0.673 mm (1.05) | 0.676 mm (1.03) |
| Range noise 1.0 mm | 0.788 mm (1.89) | **0.690 mm (1.04)** |
| 10% dropout | 0.686 mm (4.63) | **0.648 mm (1.00)** |
| 90% dropout | 1.316 mm (9.22) | **0.655 mm (1.02)** |
| Typical stereo camera | 2.652 mm (8.82) | **1.787 mm (3.67)** |

The mean barely moves on clean data. The tail moves enormously, and the tail is
what matters: a mean inside the wire diameter is perfectly compatible with one
station five millimetres out, and one station five millimetres out is the
burn-through.

**What this does not show is that the method now survives a real camera.** At
typical stereo noise it is still 1.79 mm against a 1.0 mm wire, and the range
accuracy cliff did not move at all, because no estimator averages away a
systematic accuracy limit. The honest reading is that the sweep says which sensor
class the job needs, and it agrees with what production seam trackers actually
use: laser line triangulation at a tenth of a millimetre, not general purpose
stereo at a metre.

### Welding where the joint is going to be

The measurement is cold and the weld is not. Heat put into the bottom of the seam
rotates both folded sides about their fold lines, so the top of the joint walks
inward before the torch reaches it. The thermal model predicted this from the
start, the number was computed and stored, and nothing used it: the path was
driven to the cold measurement, which is where the joint used to be.

Compensating is easy and proving it helps is not, because there is an obvious way
to cheat: predict the distortion with the same model that produces it and the
correction is exact by construction. So the cell compensates with nominal constants
while the metal moves by perturbed ones, which is the situation a real machine is
in.

| Model error | Welding the cold measurement | Welding where it will be |
| --- | --- | --- |
| -40% | 0.330 mm | 0.216 mm |
| nominal | 0.552 mm | 0.034 mm |
| +100% | 1.103 mm | 0.557 mm |

The nominal row is the case the script's own header warns about: the metal and the
compensator share one model form, so with the constants exactly right the correction is
close to exact by construction, and 0.034 mm measures very little. The off-nominal rows
are the ones that measure something. Compensating beats not compensating from -40% to +100% model error. The curve is
not symmetric and that asymmetry is the practical result: under-modelling leaves
some of the walk behind and is bounded by the distortion itself, while
over-modelling pushes the torch past the joint and is bounded by nothing. A cell
unsure how hot its own weld runs should under-correct on purpose.

### Does this only work for welding?

The loop generalises and none of the numbers do. Stated without the welding it is
three moves: measure the part rather than trusting the program, at the place the
tool will touch it; schedule the process to what was measured, station by station,
and refuse where the measurement says the process cannot work; and read the run
rather than the part, because correcting every part hides the machine that made
them wrong.

Press braking is the closest case and the strongest evidence, since high-end
brakes already close exactly this loop with angle sensors. Self-pierce riveting
needs the same stage two, with the feature a point rather than a line and the die
chosen from local stack thickness. Adhesive and sealant dispensing is nearly
isomorphic: a bead following a flange that moved, with flow rate following the gap.

What does not carry across is any number in this repository. Each process needs its
own feature to find, its own process model in place of `cell/weld_params.py`, its
own defect vocabulary and its own calibration. Claiming otherwise would be the same
mistake as quoting an accuracy figure measured on exact depth.

### What the 0.67 mm does not include

That figure is true about the tracker and would be a misrepresentation of the cell.
Published work on vision guided welding is explicit that achieved accuracy depends
on the whole chain: sensor resolution, calibration quality, standoff stability,
part surface, robot repeatability. This project measured two of those carefully and
was silent about the rest.

Systematic terms add. Independent random terms combine in quadrature. Which of the
two a term is decides whether making more parts helps, and several of these are
systematic.

| | Total | Meets 1.5 mm? |
| --- | --- | --- |
| What the seam study alone suggests | 0.68 mm | yes |
| Perfect sensor, nothing compensated | 2.23 mm | no |
| Typical stereo camera, compensated | 2.82 mm | no |
| Laser profiler, compensated | **1.16 mm** | yes |

Industrial vision guidance for weld tracking is asked for about 1.5 mm. Only the
last row meets it, which is the same conclusion the sensor sweep reached from the
other direction.

The largest single term is **hand eye calibration at 0.85 mm**: systematic, bigger
than the entire tracking error, taken from published residuals rather than measured
here, and not fixable by this cell. Every term in `cell/error_budget.py` carries
where its number came from. The first version of that file labelled its own laser
row "measured here" when it is a published figure, which is a fair demonstration of
how quietly that goes wrong.

### A drifting camera looks exactly like a drifting fixture

This one is worse than an omission, because the cell answers it confidently.

A fixture going off moves the part. A camera going off moves the measurement. Both
appear in the seam as the whole joint translating, by the same amount at the bottom
and the top, with the tilt flat. Measured on the seam alone they are not merely
similar, they are the same numbers: across four runs the discriminators differ by
0.00. So the drift diagnosis answers "fixture" to both and would send a fitter to
re-shim a fixture that is fine while the camera mount goes on moving.

No model fixes that, because the information is not in the data. What fixes it is an
instrument: a reference feature bolted to the machine base rather than to the
fixture, which the same camera reads on every part. Nothing about the press or the
fixture can move it; only the camera can.

Asked to name one of four causes, Nemotron 3 Super 120B-A12B:

| | Score | Drifting camera identified |
| --- | --- | --- |
| On the seam alone | 8 of 16 | **0 of 4** |
| With the reference | **14 of 16** | **4 of 4** |

Same model, same prompt structure, one extra column. What was missing was not
intelligence. And it is the same instrument that would catch the calibration term
above going out, so the two findings are one blind spot seen from two directions.

Once the instrument existed, the small model was retrained on the harder problem:
four causes rather than three, with the reference column, on eight hundred runs
generated away from the same held-out seeds.

| Model | Where | Four-way score | The two that look alike |
| --- | --- | --- | --- |
| Nemotron 3 Super 120B, seam only | Token Factory | 8 of 16 | 4 of 8 |
| Nemotron 3 Super 120B, with reference | Token Factory | 14 of 16 | 6 of 8 |
| Qwen3-1.7B + LoRA, with reference | the cell's own GPU | 16 of 16 | 8 of 8 |

**That 16 of 16 does not mean what it looks like, and the row is left unbolded for
that reason.** It was measured on a test set with no hard case in it. Every run in
it sits at least 0.58 away from the decision boundary and most sit one to two away,
so the answer is recoverable by comparing two numbers to a threshold. Four lines of
arithmetic with no model in them also score 16 of 16 on that set, in zero
milliseconds. A score a ruler can match is not evidence of diagnosis, and the
section below is the test built to find that out.

The three-cause result earlier in this file is kept rather than replaced. Those
models were scored on that question, and restating their numbers against a harder
one they never saw would be exactly the sort of thing the rest of this file exists
to avoid.

The honest cost is time: about 5.7 s on a laptop GPU against roughly 1.6 s for the
120B in the cloud. For a question asked once a shift on a machine that may have no
network, that is the right side of the trade, and it is a different trade from the
one the per-part decisions make.

That figure took two goes to measure. The first run reported 17 s, which was the
measurement rather than the model: the latencies fell monotonically from 39 s to
9 s across the run, which is a laptop GPU starting at a low clock and ramping. A
throwaway inference before timing, which is ordinary practice for GPU benchmarking,
brought the median to 5.7 s with the scores unchanged.

Worth being precise about where this runs, because it is the one part of the project
that does not run on Nebius. The LoRA was trained on Token Factory and the base is
pulled from Hugging Face; the merged adapter then runs on the workstation beside the
cell. Everything it is compared against is a Token Factory API call. "Trained on
Nebius, runs at the edge" is the accurate description, and no edge score in this file
should be read as a Token Factory number.

### The test the answer key passes, and what replaces it

Every drift score above it is measured on cases that are not hard. The runs were
generated by injecting a fault and checking it showed, so the closest one sits 0.58
away from the decision boundary and there is no marginal case anywhere in the set.
That has two consequences worth stating plainly. A fixed rule cannot fail on it: the
boundary written out as four lines of arithmetic scores 16 of 16 in zero
milliseconds. And the differences between models on it were largely differences in
how well each guessed a boundary the prompt never named, which is why stating the
boundary moved a frontier model from 9 of 16 to 16 of 16.

So a second set was built where a ruler and a diagnostician come apart. Cases are
generated by rejection sampling until they land in a chosen band, in five families:

| Family | What it is |
| --- | --- |
| `clear` | a fault well past the threshold, kept as a control |
| `marginal` | a real fault whose trend does **not** clear the threshold |
| `lucky_noise` | nothing injected, but scatter pushes a channel over the line |
| `subtle_camera` | a camera creeping near the line rather than far past it |
| `mixed` | a press and a fixture going out together |

Every arm is asked the same question with the same data and scored the same way. What
is allowed to differ is the envelope, because which API options an endpoint accepts is
a property of the endpoint rather than of the model's judgment, and adapting the
question per model until each looks good would not be a benchmark. Both halves of that
were learned the hard way here: one model rejects the reasoning option this project
sends and its refusals were recorded as 0 of 28, and another was addressed by an
identifier that does not exist and also came back 0 of 28. Neither is a score. A pass
whose endpoint returned errors instead of answers is dropped rather than published, and
an arm that was called without an option it refuses is marked as such.

On `mixed`, naming either genuine fault counts. Both machines need touching and
naming one sends a fitter somewhere useful; naming a cause that is not happening
does not.

Six hosted models were run on it, every one with the same prompt, the same data and
the same scoring, alongside the rule and the model running at the edge. Passes are
pooled rather than picked, and each arm's per-pass scores are shown, because the
spread matters more than the ranking:

| Arm | Score | Rate | Beats the rule on | Loses on | p | p, Holm-adjusted |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5 397B | 46/64 | **72%** | 12 | 2 | 0.013 | 0.10 |
| Nemotron 3 Ultra 550B | 45/64 | 70% | 9 | 0 | 0.004 | **0.035** |
| DeepSeek V4.1 Flash | 45/64 | 70% | 12 | 3 | 0.035 | 0.25 |
| Nemotron 3 Super 120B | 44/64 | 69% | 12 | 4 | 0.077 | 0.38 |
| DeepSeek V4 Pro | 43/64 | 67% | 8 | 1 | 0.039 | 0.25 |
| Qwen3-1.7B + LoRA, at the edge | 43/64 | 67% | 16 | 9 | 0.230 | 0.69 |
| Super 120B, boundary withheld | 38/64 | 59% | 13 | 11 | 0.839 | 0.84 |
| **The threshold as arithmetic** | 36/64 | 56% | | | baseline | |
| gpt-oss 120B | 28/64 | 44% | 4 | 12 | 0.077 | 0.38 |

Sixty-four cases, eight per family and scenario pair over eight seeds. The paired columns
compare each arm against the rule on the same cases: how many the arm got right where the
rule did not, how many the other way, and the exact McNemar p. Concordant cases say
nothing about which arm is better and are excluded. The last column corrects that p by
Holm's method over the nine arms tested against the rule: the eight here and the edge
model asked without the boundary sentence (also 43 of 64, p 0.23, adjusted 0.69).

Every arm in this table is asked the same question: the measurements, the fitted slope
of each channel against its own scatter, and the sentence that says where the boundary
is. The first version gave the cloud models all of that and gave the edge model the raw
rows alone, and put both in one table; that was two questions, not two models. Asked
the full question the edge model scores the same 43 of 64, with or without the boundary
sentence, and its training set is now written from the same text so a retrain sees what
it will be asked.

Four things fall out of it, and the first is about the test rather than the models.

**The first version of this test had a hole in it, and said nothing.** A clear camera
drift generated zero cases out of an intended eight, so one of the eight family and
scenario pairs was simply absent and the header announced "8 pairs" regardless. The
cause was a guard that looked reasonable: a case should be loud on the channel its
fault moves and quiet on the rest. But a camera moves the MEASUREMENT, so it shows on
the reference and as an apparent shift of the whole seam, and that second channel is
the entire confound the section above spends a study on. The guard threw every camera
case away. Filling the hole moved the rule's baseline from 46% to 56%, because a clear
camera is easy for a rule, so the first version also flattered every model against it.
The generator now knows which channels each fault is supposed to move, the runner
prints the count for every pair and says so loudly when one is empty, and a test
asserts that no pair is ever silently absent.

**Read the gaps against the spread and the p value, not against each other.** At
temperature zero the same model on the same cases came back 18, 16, 17 and 16 across
four passes of the earlier set. Two arms differing by one or two are not differing.
Four arms beat the rule at an uncorrected p < 0.05; after Holm correction over the nine
arms only Nemotron 3 Ultra does, at an adjusted 0.035. Nemotron 3 Super 120B, the large
Nemotron the three-cause result first pointed to, is not among the four: 69% against 56%
at p = 0.077 on its first run, and 46 of 64 on a second (`service/evidence.py`,
`hard_study()`). The cell's diagnosis default is DeepSeek V4.1 Flash, 45 of 64 at
p = 0.035, 0.25 after correction, so the default does not rest on significance. It is one
case behind the best arm and is already the model the cell calls for the bead, so the
diagnosis adds no model the cell does not already depend on. It is not the fastest: a
median 2.9 s a case against 1.5 s for Ultra. On the earlier 28 case set no arm at all
cleared p = 0.05, and an earlier draft of this section claimed the margin as a finding
anyway. Pooling repeated passes would have turned that p = 0.07 into p = 0.004, which
is significance manufactured out of a rerun; the comparison is made on one pass.

**The fine-tuned model's 16 of 16 was an artefact of the easy set.** Here it scores 67%
and answers "none" to 13 of the 16 marginal cases and to all 8 subtle camera cases (3 of
16 and 0 of 8 right). It learned the threshold and very little past it, which is exactly the behaviour an easy
set cannot distinguish from diagnosis, and its margin over the rule is not significant
(p = 0.23). It is still the right thing to run for the clear cases it was built for, at
the edge, with no network. It is not a better diagnostician than the model it was
distilled from.

**Where the models earn their place, and where the rule wins.** The gap is in the two
families a threshold cannot reach: 8 of 8 on runs with two machines drifting for every
model arm except gpt-oss (3 of 8), where the rule scores 0 of 8 because seeing two
channels move gives it no branch and it falls through to "nothing wrong", and on lucky
noise, up to 4 of 8 for the hosted models and 8 of 8 for the edge model, where the rule
scores 0 of 8 because a channel over the line is all it looks at. No model beats the
rule on subtle camera: 5 of 8 for the rule against 2 to 5 for the hosted models (gpt-oss
2), and 0 of 8 for both the edge model and the 120B with the boundary withheld. A camera creeping just past the line is exactly the case
where applying the boundary mechanically is right and reasoning around it is wrong.
Nothing here is close to answering the question, on four possible answers.

One implementation note that belongs here rather than in a commit message. On these
runs the 120B twice answered `reference`, which is a column in the data and not one
of the four causes it was asked to choose between. Scored by string comparison that
is simply wrong; scored honestly it is a different failure, a model answering a
question nobody asked. Every evaluation in this repository now routes replies through
one validator that rejects anything outside the answer space and records it as
`out_of_vocabulary`, so an unparseable reply, a refusal, and an invented category are
three distinct outcomes rather than all silently counted as a wrong diagnosis.

```bash
python scripts/eval_hard_drift.py --per-family 8
```

### Reproducing all of it

Every figure in this file is generated rather than pasted, so a stale one is a command
away from being correct:

```bash
python scripts/make_figures.py
```

The studies behind them. Most of these call models, so they need a key and are paid; the
fine-tuning and edge commands also need the edge dependencies,
`pip install -r requirements.txt -r requirements-edge.txt` (see that file for a CUDA
build of torch), and the edge scores need the adapter, which is not in git yet.

```bash
python scripts/run_seam_campaign.py --manifest data/cap50v3/manifest.json
python scripts/run_campaign.py --manifest data/cap50v3/manifest.json --no-post --routed
python scripts/eval_router_truth.py --campaign data/campaign50.jsonl
python scripts/eval_drift.py --parts 30 --runs 4
python scripts/draw_result_chart.py --campaign data/seam_campaign.jsonl

# every other model that will take a picture, on the same beads
python scripts/probe_vision_field.py
python scripts/run_inspection.py --no-post --model openbmb/MiniCPM-V-4_5 --out data/inspections_minicpm.jsonl
python scripts/run_inspection.py --no-post --model moonshotai/Kimi-K2.6 --out data/inspections_kimi.jsonl
python scripts/run_inspection.py --no-post --model zai-org/GLM-5.3-Flash --out data/inspections_glm.jsonl
python scripts/run_inspection.py --no-post --model MiniMaxAI/MiniMax-M3 --out data/inspections_minimax.jsonl
python scripts/run_inspection.py --no-post --model deepseek-ai/DeepSeek-V4.1-Flash --out data/inspections_deepseek.jsonl

# train the shift diagnosis, then score it on the cell's own GPU
python scripts/make_drift_dataset.py --train 600 --val 80
python scripts/finetune_drift.py --launch
python scripts/finetune_drift.py --watch
python scripts/finetune_drift.py --fetch
python scripts/eval_drift_edge.py
python scripts/eval_drift_edge.py --base-only     # the untrained control

# turn the faults this run found into cited corrective action briefs
python scripts/raise_briefs.py

# where the perception stops working, both arms
python scripts/eval_robustness.py --no-refine --out data/robustness_slices.jsonl
python scripts/eval_robustness.py --out data/robustness_refined.jsonl

# whether welding where the joint will be beats welding where it was
python scripts/eval_compensation.py

# the camera-versus-fixture confound, and the instrument that resolves it
python scripts/eval_confound.py

# retrain the edge model on four causes, and score it on the cell's own GPU
python scripts/make_drift_dataset.py --with-camera --train 800 --val 100 \
    --out data/finetune_camera
python scripts/finetune_drift.py --launch \
    --train data/finetune_camera/drift_train.jsonl \
    --val data/finetune_camera/drift_val.jsonl --suffix drift-four-way
python scripts/finetune_drift.py --watch
python scripts/finetune_drift.py --fetch
python scripts/eval_drift_edge.py --with-camera
python scripts/eval_drift_edge.py --with-camera --base-only
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
  post-weld capture  -------------->      DeepSeek V4.1 Flash -> pass, or a
                          |                               named defect, with reasoning
                          v
  FastAPI service + SQLite  <---------->  DeepSeek V4.1 Flash: across a run of
        |                                  parts, is the PRESS drifting or the
        |                                  FIXTURE? a press tilts the seam, a
        +--> operator dashboard            fixture shifts it
        +--> Nemotron 3.5 Lightning supervisor, reading the same rows through
             tools, and searching outside the cell for anything it cannot measure

  Ground truth exists for every part, so every number above is scored, not shown.
```

Every measurement, verdict and decision lands in one database. The dashboard and the
agent read the same rows, so what the operator sees and what the agent reasons over
cannot drift apart.

## Repository layout

```
tank/       the crawler: seam inspector, probe planner, scorers
tank/sim/   the modelled tank, the capture that turns it into a dataset with
            answers, the render look, the crawler detail pass, the seam shot
cell/       cell geometry, press forming and drift, weld thermal distortion,
            adaptive weld parameters, depth seam tracking, Isaac simulation
brain/      Nebius client, model router, view planner, inspection prompt,
            supervisor agent and its tools, outside lookup
service/    FastAPI service, database schema, API
dashboard/  operator web UI
scripts/    campaign runners, evaluations, figures, cell visualisation
docs/       platform notes, measured results, submission text
tests/      unit tests
```

The pieces that carry the argument:

| File | What it settles |
| --- | --- |
| `tank/probe_planner.py` | the coverage rule that drives the arm, and the model planner that was measured against it |
| `tank/score_probes.py` | every planner against chance and the oracle, with the tie-break check, on the same stations |
| `tank/layout_sweep.py` | how much of the sweep's score belongs to where the hidden patches were put |
| `tank/significance.py` | what each tank count licenses, with its exact bound and the paired tests |
| `tank/truth.py` | the one answer-key rule, rebuilt without Blender and checked against it |
| `tank/inspect_seam.py` | what a tank seam is judged on, and why it is not the cell's vocabulary |
| `tank/score_survey.py` | a miss, a misname and a decline counted as three different things |
| `tank/sim/capture_tank.py` | how much of a fault is in frame, asked from both sides |
| `cell/seam_track.py` | follows the seam along its length instead of fitting it straight |
| `cell/forming.py` | what a press in spec still leaves on a folded side, and what a drifting one leaves |
| `cell/thermal.py` | what the arc does to the joint while it is welding it |
| `cell/weld_params.py` | choosing the weld to suit the gap that is actually there |
| `brain/router.py` | why every per-part decision goes to the smallest Nemotron, which is not the obvious answer |
| `scripts/eval_router_truth.py` | scores every model size against ground truth rather than against each other |
| `scripts/eval_drift.py` | press or fixture, with a control |

## Setup

Requires Python 3.12. A Nebius Token Factory API key is needed only to ask a model
again; every replay, scorer and test in "If you have ten minutes" runs without one.

```bash
git clone <this-repo>
cd <repository-folder>
python -m venv .venv
source .venv/Scripts/activate    # Windows, Git Bash
# .venv\Scripts\activate         # Windows, cmd or PowerShell
# source .venv/bin/activate      # Linux and macOS
pip install -r requirements.txt
```

The edge model needs more than that, and nothing else does; `requirements-edge.txt` lists
it, with a note on installing a CUDA build of torch.

To ask models, copy the environment template and fill in your key:

```bash
cp .env.example .env
```

Everything that describes a site rather than the code is in that file: which models to
call, how long to wait on one call, what the bead camera sees (BEAD_SOURCE is
`rendering` on the simulator this repository ships with and `photograph` on a cell),
the crop box the inspector sends, and the millimetre limits every page and prompt read.
Placing the cell somewhere else is editing that file, not the code.

With a key set, check cloud access first. This makes a few small paid calls (tool calling
and one image):

```bash
python scripts/hello_nemotron.py
```

Run the service and open the dashboard at http://127.0.0.1:8000. It opens on the tank
page:

```bash
uvicorn service.app:app --reload
```

The tank page reads the recorded survey, so it is full from the start. The cell pages
start empty, and one command, run in a second terminal while the service keeps running,
fills them:

```bash
python scripts/demo.py --offline   # no key, nothing paid
python scripts/demo.py             # with a key, the half that calls models too
```

That runs the real pipeline over `data/demo`, six parts of captured depth that ship
with this repository, and posts what it finds. It does as much as it can with what
you have: seam tracking and weld scheduling need nothing but the depth frames and
finish in about three seconds, and the look-and-judge routing, the post-weld
inspection and the cited briefs each need a key. `--offline` runs only the half that
calls nothing.

The demo writes under `data/demo_run/` rather than over the campaign files in `data/`,
which are the evidence for the numbers in this README and are not the demo's to
overwrite.

Six parts is a demo, not the result. The fifty-part campaign behind the tables needs
the full capture, which is about 900 MB and is not in git:

```bash
python scripts/run_seam_campaign.py --manifest data/cap50v3/manifest.json
python scripts/run_campaign.py --manifest data/cap50v3/manifest.json --routed
python scripts/run_inspection.py --manifest data/cap50v3/manifest.json
```

### What the demo capture costs

The raw depth frames are float32, 1.6 MB each, which is why a full capture cannot be
carried in a repository. The shipped frames are quantised to uint16 over the near
field and gzipped, 49 MB of depth into 5 MB on disk. That is only allowed if the seam
that comes out is the same seam, so `scripts/make_demo_capture.py` tracks every part
from the raw frame and again from the packed one and prints the difference. The worst
station moves by **0.13 mm**, against a tracked seam accuracy of 0.68 mm and a sensor
noise floor above both.

The dashboard then shows the seam itself: the tracked path against the straight line
a taught program drives, the gap at every station, the heat the torch is putting in,
and which Nemotron answered each decision and why.

The simulation half, which produces the captures in the first place, requires Isaac
Sim 6.0 and is run with its bundled interpreter. Nothing above needs it. Capture in batches of ten rather than one run of fifty; the
Replicator graph grows as parts times frames and a single large capture stalls in
graph construction. `scripts/merge_manifests.py` stitches the batches.

## Where this fits

Robotic tank inspection is an existing industry, and this project does not claim a new
capability in it. Square Robot inspects tank floors from inside a tank that stays in
service, with a submersible robot. Gecko Robotics maps wall thickness with ultrasonic
climbing crawlers. Eddyfi's Scorpion 2 is a crawler with a dry-coupled ultrasonic wheel
probe that records shell thickness along a vertical scanline. Those descriptions are
from the vendors' own pages and were not measured here.

A probe that reads continuously as it rolls takes away the choice this project studies.
What is studied here applies to the other case: a single-element spot probe that has to
be placed, coupled and read at each point, in air, on shells where the loss worth
finding is out of sight, such as insulated or heated shells with corrosion under
insulation. The question is where a model belongs on such a crawler. The answer measured
here is on the picture, with arithmetic choosing where the probe goes.

## Prior work

General-purpose multimodal models have been benchmarked on industrial defect inspection
before. MMAD (Jiang et al., ICLR 2025, arXiv 2410.09453) put them through thousands of
questions on industrial images and found them well short of industrial requirements.
The tank inspection here is a far smaller test of the same kind, and it should be read
against its own trivial baseline: calling every frame sound scores 13 of 24 on the
walk-in call, because 13 frames hold no fault.

The joint-finding and correction concept was validated by the author in an earlier
project, published at
[vision-corrected-welding](https://github.com/harshal-96/vision-corrected-welding).
Everything in this repository was built during the hackathon submission period: the
tank crawler, its simulated survey and its scorers, the
press forming and drift model, seam tracking along the length of the joint, the weld
thermal distortion model, adaptive weld parameters from the measured gap, the view
planner that puts a reasoning model inside the control loop, the model
router and the evaluations that overturned its first design, the press-versus-fixture
diagnosis, the post-weld inspection station and defect injection, the cloud inspection
model, the supervisor agent with its tools and outside lookup, the database and
service, and the dashboard. The earlier work is the starting point, not the
submission.

## License

Apache License 2.0. See [LICENSE](LICENSE). Third-party material in the repository, the
robot arm model used to draw the welding cell, keeps its own terms, listed in
[NOTICE](NOTICE).
