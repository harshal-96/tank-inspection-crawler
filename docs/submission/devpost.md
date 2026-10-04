# Tank Inspection Crawler

A robot that crawls the seams inside a steel storage tank, judges the weld and the
plate as it goes, and decides where to spend its next ultrasonic thickness
measurement.

## Inspiration

A storage tank has to be emptied, cleaned, ventilated and certified before anyone can
climb into it, and the inspection itself is a person with a lamp and a probe standing
inside a steel drum. Confined space entry kills between ninety and a hundred and thirty
people a year in the United States alone. About a quarter of those deaths happen during
inspection, and roughly sixty per cent of the people who die are the ones who went in
afterwards to pull someone out.

The tank is out of service for two to eight weeks while it happens.

The obvious response is to send a robot, and plenty of people already do. Magnetic
crawlers with ultrasonic probes are a mature product. What is missing is not the robot.
It is the judgement: deciding, in the tank, which of the thousands of places it could
measure are the ones that matter, and which of the things it can see are worth a work
order. That decision still goes back to a person watching a screen, and it is the part
that does not scale.

## The argument, in one dataset

The run carries two patches of wall loss. One is rusted through to the surface and a
camera finds it instantly. The other has nothing on the surface at all.

The rusted one measures **inside** the acceptance limit. The invisible one does not,
and all four probe points on the run that fail the limit sit under frames the inspector
reads as sound, on plate that looks like new steel.

So neither instrument is enough by itself, and something has to decide at every station
where to spend a measurement there is only time to take twice. That decision is the
project. The rest is plumbing.

## What it does

The crawler drives a weld seam. At every station it does three things.

**It reads the picture.** A vision model judges the length of seam in frame: is the
bead sound, and if not, what is wrong with it, out of crack, burn-through, lack of
fusion, undercut, porosity and corrosion of the parent plate. It reports every fault it
can see rather than only the worst, because a frame can hold a groove in the weld and a
corroded plate at the same time and those are two different jobs for two different
crews.

**It decides where to measure.** The arm can reach seven places across the seam and
there is time for two. An NVIDIA Nemotron model picks which two, from what the camera
just reported, which offsets have gone longest without a reading, and what the last few
readings came back as.

**It refuses to guess.** A frame it cannot judge is recorded in its own column, neither
right nor wrong, as a frame somebody has to open themselves. A reply that cannot be
read never becomes a verdict.

## What it measures

Twenty four stations along four metres of circumferential seam, rendered from a
modelled tank with the answers written before any model saw a picture.

| | Result | Scored against |
| --- | --- | --- |
| Is this seam sound, or does somebody have to come and look | **24 of 24**, nothing missed, no false alarms, nothing declined | the faults actually inside the camera frustum, projected per frame |
| Naming the worst fault in the frame | 20 of 24 | the same |
| Finding every fault present, not only the worst | 9 of 13, none invented | every fault in frame, listed separately |
| Choosing where to probe, 48 measurements out of 168 places | **Nemotron 3 Nano, median 4 of 4**, worst run 2 of 4 | the true remaining wall at all 168 |
| the same, covering the band by arithmetic | 2 of 4, every single time | the same |
| the same, probing where the camera saw corrosion | 0 of 4 | the same |
| the same, always probing on the seam | 0 of 4 | the same |

The inspector runs at about 9 seconds a frame and the probe planner at about 8 seconds
a station, one model call each.

The last three rows are the point. Following the camera is worse than useless here: it
spends every measurement on the patch that is within limits and never looks at the one
that is not.

## The same loop, on a second problem

The repository also contains a self-correcting robotic welding cell, which is the
evidence that this is a method rather than one lucky demo. It photographs every part
before it welds, measures where the seam actually runs rather than where the program
assumes it will be, and raises the alarm when a machine upstream starts drifting.

| | Result |
| --- | --- |
| Seam tracked along its whole length | 22.8 mm to 0.67 mm, 50 of 50 parts under a millimetre |
| Routing each part to the right model size | 88 of 89, no part welded on a failed view |
| Post-weld bead inspection | 91% pass or defect on 150 beads, no false alarms |

Both halves share one call policy, one JSON contract, and the same rule that a parse
failure is never allowed to become a verdict.

## How it is built

Everything runs on **Nebius Token Factory** through one shared client.

| Layer | Model |
| --- | --- |
| Reading a tank seam from the crawler's camera | DeepSeek V4.1 Flash |
| Where to put the thickness probe next | NVIDIA Nemotron 3 Nano 30B-A3B |
| View planning inside the welding cell's control loop | NVIDIA Nemotron 3 Nano 30B-A3B |
| Cell supervisor, with tool calling | NVIDIA Nemotron 3.5 Lightning |
| Writing a corrective action brief from fetched documents | NVIDIA Nemotron 3.5 Lightning |
| Shift diagnosis without a network | Qwen3 1.7B with a LoRA, trained on Token Factory, run on the cell's own GPU |

The tank itself is a modelled shell: six courses, 473 metres of weld, plate thicknesses
that step down with height, a product line where the liquid used to sit, and faults
placed with their dimensions recorded before anything was rendered. The crawler's
camera and depth pass are rendered with Cycles on an RTX 4070 through OptiX.

## Challenges

**The hardest bug was always the picture, not the model.** Every time a frontier model
was marked wrong, reading its stated reason showed it had described the image
accurately and the image was wrong. Corrosion pits were modelled as spheres sitting
proud of the plate, so they were gravel and the model called them loose debris. Pores
were set inside solid weld metal where no camera could see them. Nothing in the scene
subtracts material, so a groove whose floor lay behind the sheet it was cut into was
simply hidden, and the heat-tinted band swallowed the undercut whole. The detail window
matched the crawler's run exactly, so at the last stations the camera looked past the
end of the modelled bead and the inspector correctly reported a break in the weld.

**The answer key was wrong more often than the model was.** Coverage asked how much of
a fault was inside the frame, which goes to almost nothing exactly when a corrosion
patch fills every pixel. The ground truth named wall loss for its cause while the
inspector was asked to name it for its appearance, so every correct call of rust scored
as an invention. A scoring order borrowed from new fabrication ranked plate corrosion
last, in a project whose entire premise is that plate corrosion is what opens tanks.

**Models are not interchangeable, and not in the way you expect.** Of the models on the
account, only five accept an image at all. Nemotron 3.5 Lightning runs the supervisor
perfectly well and cannot plan the probing at all: given a station it writes its
reasoning out as prose and never reaches the JSON, with three thousand tokens of room
as readily as with three hundred. Nano and Super both answer cleanly on the identical
question.

## What we are still wrong about

Undercut. Four frames carry a groove at the weld toe and the inspector reports the
corroded plate beside it instead. Asked to list everything rather than judge, it names
the groove first and describes it correctly, so it is not blind to it. Asked to judge,
it calls the toe a normal shadowed junction line, which is a fair reading of how the
groove is currently rendered. Every image-capable model on the account agrees with it,
frontier ones included, so this is a limitation of the simulated undercut and not of
the inspector. It is written up rather than tuned away.

## Why this can be deployed rather than only demonstrated

Robotic inspection has been referenced in API 653 for more than twenty five years, and
crawler data is accepted under its appendices provided the system is calibrated and the
data is interpreted by a certified inspector. The robot is a data collection tool and
human judgement on fitness for service is not replaceable.

That is exactly the shape of this system. It triages, it cites what it saw, it declines
what it cannot defend, and it hands a certified inspector a shorter and better ordered
list than they would have had. It is not trying to sign the report.

## What is next

A bench rig with a real ultrasonic thickness gauge reading a real plate with a pocket
ground thin on the back face, which is the same hidden wall loss the simulation argues
about, measured for real.
