# Tank Inspection Crawler

A robot that crawls the seams inside a steel storage tank with two instruments: a camera
that judges the weld and the plate, and an ultrasonic probe that measures what is left of
the wall behind them. The premise this project is built on, and standard API 653
practice, is that neither instrument can do the other's job. The camera cannot find
hidden wall loss, and on this rig it cannot even see 6 of the 7 places the probe can
reach. The probe cannot judge a weld surface.

What we measured is where a model belongs on that robot. The answer is on the picture.

## Inspiration

A storage tank has to be emptied, cleaned, ventilated and certified before anyone can
climb into it, and the inspection itself is a person with a lamp and a probe standing
inside a steel drum. Confined space entry is dangerous work: US Bureau of Labor
Statistics figures put confined-space deaths at roughly 90 to 130 a year, and NIOSH has
reported that a large share of the people who die are would-be rescuers.

A tank comes out of service for its internal inspection anyway, mostly for the floor.
The crawler does not remove that outage. What it can cut is the time people spend inside
the tank while it is open.

## Where this fits

Robotic tank inspection is an established field. Gecko Robotics, Eddyfi (the Scorpion 2
crawler) and Diakont build crawlers that scan walls and floors densely, and pairing a
fast screening instrument with ultrasonic follow-up is established practice. Square Robot
and the PetroBot programme work on inspecting tanks without people inside. On a bare,
uninsulated shell, wall thickness is usually better measured from the outside while the
tank stays in service.

This project is aimed at a narrower case: a crawler carrying a single-element spot
probe, where every reading costs time to position, couple and settle, on shells that
cannot easily be read from outside, such as insulated or heated tanks where corrosion
under insulation leaves nothing on the surface the crawler can see. On that robot the
question is which part of the job a model should do.

## The argument, in one dataset

The survey run carries six patches of wall loss. One shows on the surface as rust, and a
camera finds it at once; it has 69 percent of its wall left and passes. The other
five leave no mark; the worst has 34.5 percent left and fails, against a 60 percent
screening limit set for this benchmark (a screening threshold, not an API 653 minimum
thickness, which depends on the tank's fill height, product and joint efficiency). Of
the 168 places the probe can reach, 28 fail, and 20 of those 28 sit at stations whose
frame the inspector passes.

The scene is built that way on purpose. It encodes the premise, from field practice,
that the loss worth finding often leaves no mark on the surface you can see. What the
run measures is how each planner copes with it.

The geometry matters too. The camera's frame covers about 0.166 m across the seam, so of
the probe's seven offsets (0, 0.15, 0.30 and 0.45 m either side) it only ever images the
one on the seam. And the camera sits 0.36 m ahead of the probe, so the picture and the
reading at a station are not of the same piece of plate.

So something has to decide, at every station, where to spend two measurements out of
seven. Who should make that decision is what this run measures.

## What it does

The crawler drives a weld seam. At every station it does three things.

**It reads the picture.** A vision model judges the length of seam in frame: is the
bead sound, and if not, what is wrong with it, out of crack, burn-through, lack of
fusion, undercut, porosity and corrosion of the parent plate. It reports every fault it
can see rather than only the worst, because a frame can hold a groove in the weld and a
corroded plate at the same time and those are two different jobs for two different
crews.

**It decides where to measure.** The arm can reach seven places across the seam and
there is time for two. That budget is a stated design assumption for a spot probe that
is placed, coupled and read at each point, not a measured cycle time. A coverage rule picks them: chase a low reading from the last two
stations, otherwise take the offset that has gone longest without one. We also asked
NVIDIA Nemotron 3 Nano to choose them. It found 9 to 12 of 28 against the rule's 16 and
reached 4 of 5 hidden patches against 5 of 5. Given only the state the model sees, the
same rule finds 11, level with the model. So the model was never ahead, and on matched
information the two are level. The rule drives the arm because it is no worse, it is
deterministic, it costs nothing and it adds no hours.

**It refuses to guess.** A frame it cannot judge is recorded in its own column, neither
right nor wrong, as a frame somebody has to open themselves. A reply that cannot be read
never becomes a verdict. On this run nothing was declined, and the prompt only allows a
decline when no bead is in frame, so that column is empty by construction here.

## What it measures

Twenty four stations along four metres of circumferential seam, rendered from a modelled
tank. The answer key is computed from the scene, not from any model's answer. The
inspector is DeepSeek V4.1 Flash; its recorded verdicts are re-marked by
`python tank/score_survey.py --replay`, and `python tank/significance.py` prints the
bounds.

| | Result | One sided 95% exact lower bound |
| --- | --- | --- |
| Is this seam sound, or does somebody have to walk in | **23 of 24**; 1 missed, 0 false alarms, 0 declined | 0.817 |
| The same call, counted only when the fault it names is actually in view | 21 of 24 | 0.708 |
| Naming the worst fault in the frame | 19 of 24 | 0.611 |
| Finding every fault present, not only the worst | 9 of 14, none invented | 0.390 |
| Frames with every fault found | 19 of 24 | 0.611 |
| Distinct faults on the run named in at least one frame that holds them | 4 of 5 | 0.343 |

The one miss is station 12, which holds the same 64 mm crack that frame 11 had already
named. The fault never named is the undercut in frames 1 to 4, reported each time as the
corroded plate beside it. In frames 1 and 2 that rust is a sliver below the counting
threshold, so those two gate calls were right for the wrong reason, which is why the
strict row is 21 and not 23. These are not independent trials: a frame shows 0.295 m of
seam and the stations are 0.1739 m apart, so 41 percent of each frame is in the next
one, and 14 sightings are 5 physical faults. The 24 frames are also a development set:
the scene was corrected after reading model answers on these same frames.

Choosing where to probe, 48 measurements out of 168 places, scored against the true
remaining wall at all 168. `python tank/score_probes.py --baselines` prints the
arithmetic rows and writes nothing; `python tank/significance.py` scores the recorded
model runs. `python tank/score_probes.py --tiebreaks` prints the tie-break row and
`python tank/layout_sweep.py` the layout figures below; neither calls a model or writes
a file.

| Planner | Rejectable found, of 28 | Hidden patches reached, of 5 | Chance a random planner does as well |
| --- | --- | --- | --- |
| Oracle, knows the answers | 23 | | |
| Coverage rule, shipped tie-break order | **16** | 5 | 0.0002 |
| The same rule under all 5040 tie-break orders | 7 to 18, median 12 | | |
| The same rule, given only the state the model sees | 11 | 5 | 0.11 |
| Nemotron 3 Nano, three runs | 9, 10 and 12 | 4 each | 0.40, 0.23, 0.047 |
| Nemotron 3 Nano, camera report removed | 11 | 4 | 0.11 |
| Two of seven at random | 8 expected, central 95% 4 to 12 | | |
| Always on the seam, then -0.15 m | 2 | 1 | 0.9997 |
| Probe beside reported rust | 2 | 1 | 0.9997 |

The rule found more than the model in every recorded run, but the exact paired test per
run gives p = 0.14 to 0.34 (camera hidden, 0.27), so no run is significant on its own,
and the points cluster in five patches, so the pairs are not independent either. For
three of the four model runs (9, 10 and 11 found), a random picker does as well at least
11 percent of the time (P 0.40, 0.23, 0.11). The fourth, 12, is at the top edge of the
random band (P 0.047). The rule's 16 is a favourable
draw: 93.9 percent of the other tie-break orders find fewer than 16, and on 1000
random layouts of the same five patches the shipped layout suits it better than 94.9
percent of them. On a random layout the rule finds a median of 41 percent of the
rejectable points, and finds more than always-on-the-seam in only 572 of 1000.

On this run the "probe beside reported rust" baseline finds the same 2 points as
always-on-the-seam. Under the frames the
inspector flagged there are only 8 rejectable points, and with two probes a station only
6 of those can be reached. With the camera's report removed from its prompt, the model
found 11, inside the range of its three sighted runs. That is one run, so read it as no
change seen, and since the camera cannot see 6 of the 7 offsets, it could not have shown
much either way.

The inspector runs at about 9.2 seconds a frame and the probe planner at about 7.1
seconds a station, one model call each.

## The same loop, on a second problem

The repository also contains a self-correcting robotic welding cell, which is the
evidence that this is a method rather than one lucky demo. It photographs every part
before it welds and measures where the seam actually runs rather than where the program
assumes it will be, because camber, springback and fit-up move the joint from one part
to the next. It also raises the alarm when a machine upstream starts drifting. This is
where the NVIDIA models do live work: Nemotron 3 Nano decides on every part whether the
view is good enough to weld on, and Nemotron 3.5 Lightning runs the supervisor and
writes the corrective action briefs.

| | Result |
| --- | --- |
| Seam tracked along its whole length | 22.8 mm to 0.67 mm, 50 of 50 parts under a millimetre (arrival spread exaggerated for visibility, part variation on top). That is the tracker on exact rendered depth; with hand-eye calibration and the rest of the chain it comes to 2.82 mm against a 1.5 mm requirement, which this cell cannot yet measure |
| Deciding whether a view is good enough to weld on | 88 of 89; the one miss accepted a view a second look would have improved by 0.3 mm |
| Post-weld bead inspection | 91% pass or defect on 150 beads, no false alarms |
| Supervisor, on questions its own database answers exactly | 9 of 10 |
| Corrective action briefs: back, cited, figure carried, steps | 6 of 6 on all four |

Both halves share one call policy, one JSON contract, and the same rule that a parse
failure is never allowed to become a verdict.

## How it is built

Everything runs on **Nebius Token Factory** through one shared client.

| Layer | Model |
| --- | --- |
| Reading a tank seam from the crawler's camera | DeepSeek V4.1 Flash |
| Where to put the thickness probe next | NVIDIA Nemotron 3 Nano 30B-A3B, measured; the coverage rule is used |
| Per-part view decisions inside the welding cell's control loop | NVIDIA Nemotron 3 Nano 30B-A3B |
| Cell supervisor, with tool calling | NVIDIA Nemotron 3.5 Lightning |
| Writing a corrective action brief from fetched documents | NVIDIA Nemotron 3.5 Lightning |
| Shift diagnosis without a network | Qwen3 1.7B with a LoRA, trained on Token Factory, run on the cell's own GPU |

The tank itself is a modelled shell: six courses, 473 metres of weld, plate thicknesses
that step down with height, a product line where the liquid used to sit, and faults
placed with their dimensions recorded before anything was rendered. The weld bead is
modelled at about 2.8 times the API 650 reinforcement limit for this joint, exaggerated
for visibility. The crawler's camera and depth pass are rendered with Cycles on an RTX
4070 through OptiX.

## Challenges

**The hardest bug was usually the picture, not the model.** When the model was marked
wrong, reading its stated reason often showed it had described the image accurately and
the image was wrong. Corrosion pits were modelled as spheres sitting proud of the plate,
so they were gravel and the model called them loose debris. Pores were set inside solid
weld metal where no camera could see them. Nothing in the scene subtracts material, so a
groove whose floor lay behind the sheet it was cut into was simply hidden, and the
heat-tinted band swallowed the undercut whole. The detail window matched the crawler's
run exactly, so at the last stations the camera looked past the end of the modelled bead
and the inspector correctly reported a break in the weld. Every one of those fixes was
made after reading answers on these frames, which is why we call them a development set.

**The answer key was wrong more often than the model was.** Coverage asked how much of a
fault was inside the frame, which goes to almost nothing exactly when a corrosion patch
fills every pixel. The ground truth named wall loss for its cause while the inspector was
asked to name it for its appearance, so every correct call of rust scored as an
invention. A scoring order borrowed from new fabrication ranked plate corrosion last, in
a project whose entire premise is that plate corrosion is what opens tanks.

One more cost us a clean score. The key placed every fault at the crown of the bead, 8 mm
proud of the plate. The crack is drawn 2.4 mm proud and shows on the plate above and
below the bead, and near the edge of the lens that one difference put it just outside
frame 12, a frame it is plainly in. The verdict had been recorded against "crack", and
the replay had silently overwritten that with "none". The key now projects each seam fault from its rendered vertices, is
checked against Blender's own camera projection on 630 points, and refuses to rebuild
unless every frame matches its recorded hash. Station 12 is now the one miss in the table
above.

**Models are not interchangeable, and not in the way you expect.** Of the 23 models the
key reaches, 15 refused an image outright, all four NVIDIA models among them; 8 accepted
it, and 6 of those judged the bead. The Nemotron sizes do not even agree on where the
answer goes: Lightning fills `content`, while Nano returns `content` empty and puts the
whole answer, JSON and all, in `reasoning`. Read only `content` and Nano silently scores
as whatever your fallback says. And the documented structured output controls,
`guided_json` and `response_format`, had no effect on Lightning; only
`reasoning_effort="none"` stopped it reasoning in prose before the answer.

**The headline we nearly submitted was wrong, and finding that out is the best thing
in this project.**

The run originally held four rejectable probe points. On four points Nemotron Nano
found three or four and the arithmetic rule found two, so the model appeared to win.
We wrote that down. Then we ran the paired test on it: p = 0.50 to 1.00 across the three
runs. No evidence in either direction. Four points cannot separate two planners.

So the evaluation scene was given five patches of hidden wall loss instead of one,
taking it from 4 rejectable points to 28. Nothing visible changed, because loss a camera
cannot see adds no geometry, and the 24 scored frames stayed byte for byte identical.

On 28 points the rule was ahead in every run, 16 against 9, 10 and 12. But the model was
shown less state than the rule (its last six readings, with no record of how long each
offset had gone unmeasured), and the hidden patches were hand-placed after the first
result, all at least 0.27 m off the seam, where a coverage rule does well. Given only
what the model sees, the rule finds 11. So the size of the gap belongs to this scene.
Nothing here shows a model choosing probe points better than coverage, and nothing shows
coverage reasoning better than the model either; what the gap mostly measures is the
state the planner keeps. That fits published work on language-model planning, such as
PlanBench (Valmeekam et al.) and the LLM-Modulo framework (arXiv 2402.01817), which find
that language models are weak planners on their own and more useful inside a loop with
an external checker. The standard tool for this job is kriging or Gaussian-process
thickness mapping, which we have not run here.

Arithmetic drives the arm because on this evidence it is no worse, it is deterministic
and free, and a model planner would add about 5.3 hours to a real 473 m tank (4.0 even
overlapped with driving), because each plan has to wait for the last station's readings.
The model belongs on the picture, where there is something to judge.

## What we are still wrong about

- **Undercut.** Four frames carry a groove at the weld toe and the inspector reports the
  corroded plate beside it every time. That is the recorded fact. Whether the cause is
  the model or the way the groove is rendered has not been separated, and it is written
  up rather than tuned away.
- **The crack at station 12.** The inspector named it in frame 11 and passed it in frame
  12.
- **The world is ours.** We built the tank, planted the faults and wrote the answers, and
  the evaluation scene carries more hidden loss than a real shell of this age would.
  Every number here is a statement about this scene, on a development set, from one
  inspection pass.
- **Hardware assumptions.** The probe's cross-slide is notional. The magnetic adhesion
  envelope assumes 220 N of rated pull a wheel; to hold a safety factor of 3 in the worst
  dry case each wheel needs about 435 N, and with couplant on the wheels it falls below a
  safety factor of 3 in every case, so the wheels have to be kept out of the couplant or
  rated higher.
- **Speed.** A 473 m shell is about 1.3 hours of driving and 7.0 hours of vision
  inference run serially. About six requests in flight would keep inference at drive
  speed; nothing here does that yet. Cost per tank has not been measured.

## Why this can be deployed rather than only demonstrated

API 653 and API RP 575 recognise robotic in-service inspection of tank floors, and shell
wall thickness from any crawler is ordinary UT data interpreted by an authorised API 653
inspector. The robot is a data collection tool, and human judgement on fitness for
service is not replaceable.

That is the shape of this system. It triages, it cites what it saw, it records what it
cannot judge, and it hands an authorised inspector a shorter and better ordered list than
they would have had. It is not trying to sign the report.

## What is next

1. A bench plate built to `docs/bench/fabrication_spec.md`: real weld defects on the
   front face and two flat-bottomed pockets milled into the back, leaving 3.0 mm and
   4.0 mm of metal, so the hidden wall loss the simulation argues about is measured for
   real.
2. Logging a real ultrasonic thickness gauge into the same scoring, against the
   machinist's measured figures.
3. Running the vision inference concurrently while the crawler drives, so it keeps up
   with drive speed.
4. Cosmos on Nebius AI Cloud, which Token Factory does not serve. No result from it is
   claimed here.
