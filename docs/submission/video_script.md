# Video script and shot list

Target 2:45, hard limit 3:00. The Physical AI track needs at least one minute showing
the key application modules in action, so the block from 0:40 to 1:45 is the one that
must not be cut. It is 65 seconds of the repository's own commands running.

Record at 1920x1080, 24 fps, to match the rendered footage.

---

## 0:00 to 0:18  The problem

**On screen.** `tank/data/shots/seam_frames` rendered shot, held wide at the start, the
crawler small against the shell. Slow. Let it breathe.

**Narration.**
> A storage tank has to be emptied, cleaned, ventilated and certified before anyone can
> climb inside it. Then a person goes in with a lamp and a probe.
>
> Confined space work is dangerous. US Bureau of Labor Statistics figures put the deaths
> at roughly ninety to a hundred and thirty a year, and NIOSH has reported that a large
> share of those who die are would-be rescuers.

**Lower third, plain text, no animation.** Roughly 90 to 130 US confined-space deaths a
year (BLS figures).

---

## 0:18 to 0:40  What this crawler is for

**On screen.** Cut to `crawler_hero.png`, then push slowly into the macro frame
`hero_v2.png` so the weld fills the screen.

**Narration.**
> The tank is drained for its floor inspection anyway. This crawler does not remove that
> outage. It cuts the hours people spend inside.
>
> It carries two instruments. A camera that judges the weld, and an ultrasonic probe that
> measures the wall behind it. Neither can do the other's job. So the question is which
> part of the work a model should do.

---

## 0:40 to 1:45  The modules running

This is the required minute. Everything in it runs from the repository with no key: the
recorded inspector verdicts re-marked against the answer key, and the arithmetic
planners run. Caption each terminal clip with the command, so nobody mistakes a replay
for a live call.

### 0:40 to 0:52  The run

**On screen.** The full ten second seam shot, uncut, from
`tank/data/shots/seam_shot.mp4`.

**Narration.**
> This is the crawler working a circumferential seam, rendered from a modelled tank. The
> answer key is computed from the scene, not from any model's answer.

### 0:52 to 1:12  Reading the seam

**On screen.** Screen recording of a terminal running
`python tank/score_survey.py --replay`, captioned "recorded verdicts, re-marked". Hold
on the frame table, then on the summary lines.

**Narration.**
> At every station a vision model on Nebius Token Factory judged the length of seam in
> frame. This replays its recorded verdicts against the answer key.
>
> Twenty three of twenty four on the call that decides whether a person has to walk in.
> No false alarms. One miss: a crack the frame before had already named.

**Lower third.** pass or defect 23/24 . missed 1 . false alarms 0 . replayed from
recorded verdicts

### 1:12 to 1:45  The decision the camera cannot make

**On screen.** A still, held, split left and right: frame 5, the rusty patch, and
frame 15, plate the inspector passes. Caption frame 15: "the probe offsets that fail
here, 0.30 and 0.45 m below the seam, are outside this frame". Then cut to a terminal
running `python tank/score_probes.py --baselines`, captioned "arithmetic planners, no
model", and hold on the table. Then `python tank/significance.py`, held on the Nemotron
rows.

**Narration.**
> The run carries six patches of wall loss. One shows as rust on the surface and the
> camera finds it at once. It is inside the limit. The other five leave no mark, and they fail.
>
> Twenty eight of the hundred and sixty eight places the probe can reach fail the limit,
> and twenty of those twenty eight sit under frames the inspector passes.
>
> There is time for two measurements in seven at each station. The coverage rule finds
> sixteen. Nemotron Nano, asked to plan it, found nine to twelve. Give the rule only what
> the model is shown, and it finds eleven.
>
> So the model was never ahead, and on the same information they are level. Arithmetic
> drives the arm because it is no worse, it is deterministic, it is free, and it adds no
> hours. The model belongs on the picture.

**Lower third, as the tables land.** coverage rule 16/28 . same rule, model's view
11/28 . Nemotron Nano 9 to 12/28 . random picks 4 to 12

---

## 1:45 to 2:10  What it runs on

**On screen.** The model table from the "How it is built" section of
`docs/submission/devpost.md`, as plain text, each row highlighted as it is named. Then
the `reply_text` docstring in `brain/nebius.py`. No logos, no spinning cubes.

**Narration.**
> Everything runs on Nebius Token Factory through one shared client. DeepSeek V4.1 Flash
> reads the seam. Nemotron Nano plans views in the welding cell. Nemotron Lightning
> supervises and writes the corrective action briefs. Nano was measured as the tank's
> probe planner and was never ahead of arithmetic, so the arithmetic is what runs.
>
> One thing worth saying out loud. The Nemotron sizes do not agree on where the answer
> goes. Lightning puts it in the content field. Nano leaves that empty and puts the whole
> answer in the reasoning field. Read only one, and an evaluation quietly scores the
> fallback.

---

## 2:10 to 2:32  The same loop on a second problem

**On screen.** `docs/figures/seam_result.png`, the before and after scatter. Then one
or two frames of the welding cell.

**Narration.**
> The same loop was first measured on a different problem. A welding cell that
> photographs every part before it welds and measures where the seam actually runs,
> because part variation moves it. The tracker found the joint to under a millimetre on
> fifty of fifty simulated parts, on exact rendered depth. With calibration and the rest
> of the chain it comes to about 2.8 millimetres, which this cell cannot yet measure.
>
> That is the point. This is a method, not one demonstration.

**Lower third.** 22.8 mm to 0.67 mm . 50 of 50 parts under a millimetre . tracker,
exact rendered depth . whole chain 2.82 mm against 1.5 mm . arrival spread exaggerated
for visibility

---

## 2:32 to 2:45  What is still wrong, and what is next

**On screen.** The undercut frame, held. Then black with the closing line.

**Narration.**
> One honest thing. There is a groove at the weld toe on four of these frames, and the
> inspector reports the corroded plate beside it every time. Whether that is the model
> or the rendered groove, we have not separated. It is written up rather than tuned away.
>
> Next is a bench plate with two pockets milled into its back face, and a real gauge.
> The same hidden wall loss, measured for real.

---

## Capture checklist

Record these before editing. Everything is runnable with no key, and none of these
commands writes a file.

| Clip | Command or file | Length |
| --- | --- | --- |
| The seam shot | `tank/data/shots/seam_shot.mp4` | 10 s |
| Crawler hero still | `tank/data/shots/crawler_hero.png` | still |
| Macro weld still | `tank/data/shots/hero_v2.png` | still |
| Inspection, replayed | `python tank/score_survey.py --replay` | seconds; hold on the summary |
| Probe planning, arithmetic | `python tank/score_probes.py --baselines` | seconds; hold on the table |
| Probe planning, recorded model runs | `python tank/significance.py` | seconds; hold on the Nemotron rows |
| Rust frame and passed frame | `tank/data/survey/frames/rgb_0005.png` and `rgb_0015.png` | stills |
| Undercut frame | `tank/data/survey/frames/rgb_0003.png` | still |
| Cell result chart | `docs/figures/seam_result.png` | still |

Do not record a `--live` run. If one is ever needed for a clip, pass `--out` to a scratch
path outside the repository. Commit the working tree before recording. After recording,
`git diff --exit-code HEAD -- tank/data/survey` must exit 0 and
`git status --porcelain -- tank/data/survey` must print nothing; if either fails,
something wrote over the record and the take is void.

Terminal settings for the recordings: dark background, a font large enough to read at
1080p without zooming, and a window about 110 columns wide so the score table does not
wrap. Clear the screen before each run so nothing from a previous attempt is on screen.

## Rules check

- Under three minutes.
- 65 seconds of modules in action, which satisfies the Physical AI requirement for a
  project with no hardware component.
- Every number spoken appears on screen and is printed by a command in the repository:
  23 of 24, 1 missed and 0 false alarms by `score_survey.py --replay`; 168, 28, 20, 16
  and 11, and the five hidden patches, by `score_probes.py --baselines`; 9 to 12 by
  `significance.py`; the cell figures by `scripts/run_seam_campaign.py`, and the 2.82 mm
  chain by `cell/error_budget.py` (its `summary()`, pure arithmetic, set out in the
  README section "What the 0.67 mm does not include"). The confined-space figure is a public BLS figure,
  shown with its source.
- No claim is made about hardware that does not exist.
