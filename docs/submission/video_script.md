# Video script and shot list

Target 2:45, hard limit 3:00. The Physical AI track needs at least one minute showing
the key application modules in action, so the block from 0:40 to 1:45 is the one that
must not be cut. It is 65 seconds of things actually running.

Record at 1920x1080, 24 fps, to match the rendered footage.

---

## 0:00 to 0:18  The problem

**On screen.** `tank/data/shots/seam_frames` rendered shot, held wide at the start, the
crawler small against the shell. Slow. Let it breathe.

**Narration.**
> A storage tank has to be emptied, cleaned, ventilated and certified before anyone can
> climb inside it. Then a person goes in with a lamp and a probe.
>
> Confined space entry kills between ninety and a hundred and thirty people a year in
> the United States. About a quarter of those deaths happen during inspection. Around
> sixty per cent of the people who die went in to pull someone else out.

**Lower third, plain text, no animation.** 90 to 130 deaths a year. A quarter during
inspection.

---

## 0:18 to 0:40  What is actually missing

**On screen.** Cut to `crawler_hero.png`, then push slowly into the macro frame
`hero_v2.png` so the weld fills the screen.

**Narration.**
> Robots already do this. Magnetic crawlers with ultrasonic probes are a mature
> product. What is missing is not the robot.
>
> It is the judgement. Which of the thousands of places it could measure are the ones
> that matter, and which of the things it can see are worth a work order. That still
> goes back to a person watching a screen, and that is the part that does not scale.

---

## 0:40 to 1:45  The modules running

This is the required minute. Everything in it is live, not a mock.

### 0:40 to 0:52  The run

**On screen.** The full ten second seam shot, uncut, from
`tank/data/shots/seam_shot.mp4`.

**Narration.**
> This is the crawler working a circumferential seam. One metre of travel, at the speed
> a magnetic crawler actually moves.

### 0:52 to 1:12  Reading the seam

**On screen.** Screen recording of a terminal running `python tank/score_survey.py`.
Let the rows print in real time. Do not speed it up: the nine seconds a frame is part
of the claim. Cut away once eight or ten rows are down, then hold on the summary line.

**Narration.**
> At every station a vision model on Nebius Token Factory judges the length of seam in
> frame, against answers written before it ever saw a picture.
>
> Twenty four of twenty four on the call that decides whether a person has to walk in.
> Nothing missed. No false alarms. Nothing declined.

**Lower third.** pass or defect 24/24 . missed 0 . false alarms 0

### 1:12 to 1:45  The decision the camera cannot make

**On screen.** A still, held, split left and right: the frame of the rusty patch and
the frame of the clean plate above the hidden pocket. Then cut to a terminal running
`python tank/score_probes.py` and hold on the comparison table.

**Narration.**
> Here is the whole argument in one dataset. Two patches of wall loss. One is rusted
> through to the surface and the camera finds it instantly. The other has nothing on
> the surface at all.
>
> The rusted one is inside the acceptance limit. The invisible one is not. Every probe
> point on this run that fails the limit sits under a frame the inspector reads as
> sound.
>
> So something has to choose where to measure. There are a hundred and sixty eight
> places the arm can reach, time for forty eight of them, and twenty eight of them
> fail the limit. Covering the band by arithmetic finds sixteen. Asking a model to
> plan it finds nine to twelve. Probing where the camera saw corrosion finds none.
>
> We expected the model to win. It does not, and that is worth saying out loud. The
> loss is sparse and the budget is fixed, and under those conditions coverage beats
> cleverness. The model earns its place on the picture, not on the arm.

**Lower third, as the table lands.** arithmetic 16/28 . Nemotron Nano 9 to 12/28 .
follow the camera 0/28

---

## 1:45 to 2:10  What it runs on

**On screen.** The model table from the README, on screen as plain text, each row
highlighted as it is named. No logos, no spinning cubes.

**Narration.**
> Everything runs on Nebius Token Factory through one shared client. A vision model
> reads the seam. NVIDIA Nemotron Nano plans the probing and the view planning.
> Nemotron Lightning runs the supervisor and writes the corrective action briefs. A
> small fine-tuned model runs on the cell's own GPU for when the network is gone.
>
> One thing worth saying out loud. Nemotron Lightning runs the supervisor perfectly
> well and cannot plan the probing at all. It writes its reasoning out as prose and
> never reaches the structured answer, at any token budget. Nano does it immediately.
> They are not interchangeable, and the difference is not size.

---

## 2:10 to 2:32  The same loop on a second problem

**On screen.** `docs/figures/seam_result.png`, the before and after scatter. Then one
or two frames of the welding cell.

**Narration.**
> The same loop was already measured on a different problem. A welding cell that
> photographs every part before it welds and measures where the seam actually runs.
> Parts arriving centimetres out of position, welded to under a millimetre, fifty out
> of fifty.
>
> That is the point. This is a method, not one demonstration.

**Lower third.** 22.8 mm to 0.67 mm . 50 of 50 parts under a millimetre

---

## 2:32 to 2:45  What is still wrong, and what is next

**On screen.** The undercut frame, held. Then black with the closing line.

**Narration.**
> One honest thing. There is a groove at the weld toe on four of these frames and the
> inspector reports the corroded plate beside it instead. Every image model on the
> account agrees with it, so that is the simulated groove at fault and not the model.
> It is written up rather than tuned away.
>
> Next is a bench rig with a real gauge on a real plate with a pocket ground thin on
> the back face. The same hidden wall loss, measured for real.

---

## Capture checklist

Record these before editing. Everything is already runnable.

| Clip | Command or file | Length |
| --- | --- | --- |
| The seam shot | `tank/data/shots/seam_shot.mp4` | 10 s |
| Crawler hero still | `tank/data/shots/crawler_hero.png` | still |
| Macro weld still | `tank/data/shots/hero_v2.png` | still |
| Inspection running | `python tank/score_survey.py` | about 4 min, cut to 20 s |
| Probe planning running | `python tank/score_probes.py` | about 4 min, cut to 25 s |
| Rust frame and clean frame | `tank/data/survey/frames/rgb_0005.png` and `rgb_0015.png` | stills |
| Undercut frame | `tank/data/survey/frames/rgb_0003.png` | still |
| Cell result chart | `docs/figures/seam_result.png` | still |

Terminal settings for the recordings: dark background, a font large enough to read at
1080p without zooming, and a window about 110 columns wide so the score table does not
wrap. Clear the screen before each run so nothing from a previous attempt is on screen.

## Rules check

- Under three minutes.
- 65 seconds of modules in action, which satisfies the Physical AI requirement for a
  project with no hardware component.
- Every number spoken appears on screen and is reproducible from the repository.
- No claim is made about hardware that does not exist.
