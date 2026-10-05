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
> A storage tank has to be emptied, cleaned, ventilated and certified before anyone
> climbs inside. Then a person goes in with a lamp and a probe.
>
> Confined space work kills roughly ninety to a hundred and thirty people a year in the
> US.

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
> It carries two instruments: a camera that judges the weld, and an ultrasonic probe that
> measures the wall behind it. Neither can do the other's job. Which part should a model
> do?

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
> A crawler working a circumferential seam, the welded line where two rings of plate
> meet, in a modelled tank. The answer key comes from the scene, not from a model.

### 0:52 to 1:12  Reading the seam

**On screen.** Screen recording of a terminal running
`python tank/score_survey.py --replay`, captioned "recorded verdicts, re-marked". Hold
on the frame table, then on the summary lines.

**Narration.**
> At every station a vision model on Nebius Token Factory judged the seam in frame.
> Twenty three of twenty four right on whether a person has to walk in, on a development
> set. It passed all thirteen clean frames. The one miss is a crack the frame before had
> already named.

**Lower third.** pass or defect 23/24, development set . clean frames passed 13/13 .
fault frames flagged 10/11 . missed 1 . false alarms 0 . replayed from recorded verdicts

### 1:12 to 1:45  The decision the camera cannot make

**On screen.** A still, held, split left and right: frame 5, the rusty patch, and
frame 15, plate the inspector passes. Caption frame 15: "the probe offsets that fail
here, 0.30 and 0.45 m below the seam, are outside this frame". Then cut to a terminal
running `python tank/score_probes.py --baselines`, captioned "arithmetic planners, no
model", and hold on the table. Then `python tank/score_probes.py --tiebreaks`, held on
the "mean 12.35, median 12" line, and `python tank/significance.py`, held on the Nemotron
rows and their paired p.

**Narration.**
> One patch of wall loss shows as rust and is inside the limit. Five more leave no mark,
> and fail.
>
> At each station there is time for two measurements in seven. The coverage rule found
> sixteen at its shipped settings, typically twelve over every tie-break order. Nemotron
> Nano found nine to twelve. No run is a significant difference, and the hidden patches
> were placed where coverage does well.
>
> Arithmetic drives the arm: no worse, deterministic, free. The model belongs on the
> picture.

**Lower third, as the tables land.** 28 of 168 places fail, on 5 hidden patches . 20 of
them under frames the inspector passed . coverage rule 16/28 at its shipped tie-break,
median 12 over all 5040 . Nemotron Nano 9 to 12/28 . no run significant, paired p 0.143
to 0.344 . random picks 4 to 12

---

## 1:45 to 2:10  What it runs on

**On screen.** The model table from the "How it is built" section of
`docs/submission/devpost.md`, as plain text, each row highlighted as it is named, with
the lower third below. Then, for the last 9 seconds, the live clip below if the owner has
approved and recorded it, with no narration over it. No logos, no spinning cubes.

**Live clip, about 9 seconds, owner's take, paid.** One short live run, recorded only
with the owner's approval because it spends real Token Factory requests:
`python scripts/demo.py` with a key set, which runs `scripts/run_campaign.py --routed`,
held on the per-part lines as the Nemotron 3 Nano view decisions for the six demo parts
come back. Those lines show the part, the error before and after, the number of looks, the
spread and the start of the decision's reason; the per-call latency is not printed, it
is in `data/demo_run/campaign.jsonl` under `latency_ms`. It needs the service running,
and with a key it also judges the six beads with DeepSeek V4.1 Flash (and writes briefs
if a Tavily key is set), so budget the whole run, not only the Nano calls. It writes
under `data/demo_run/`, which is gitignored. Caption it on screen for its whole length:
"live call: Nemotron 3 Nano on Nebius Token Factory, not a replay". Every scored number
in this video still comes from the replays; the clip shows the call running, it does not
score anything. If it is not recorded, hold on the table for those 9 seconds instead.

**Narration.**
> Every hosted model call runs on Nebius Token Factory. DeepSeek V4.1 Flash reads the
> seam. In the welding cell, Nemotron Nano makes every per-part view decision, and
> Nemotron Lightning supervises and writes the briefs. In the tank, the arithmetic runs.

**Lower third, over the table.** Nemotron 3 Nano, per-part view decision 88/89 . the
prompt's limits as arithmetic 86/89, paired p 0.625 . Nano's 3 extra right calls cite a
misread inlier balance . tank probe planner: the coverage rule . offline diagnosis model
trained on Token Factory, runs on the cell

---

## 2:10 to 2:32  The same loop on a second problem

**On screen.** `docs/figures/seam_result.png`, the before and after scatter. Then one
or two frames of the welding cell.

**Narration.**
> The same loop first ran on a welding cell that finds where each part's joint runs,
> because part variation moves it. On fifty of fifty simulated parts its tracker is under
> a millimetre, on exact rendered depth. The whole chain comes to about 2.8 millimetres,
> which this cell cannot yet measure.
>
> A method, not one demonstration.

**Lower third.** 22.8 mm to 0.67 mm . 50 of 50 parts under a millimetre . tracker,
exact rendered depth . whole chain 2.82 mm against 1.5 mm . arrival spread exaggerated
for visibility

---

## 2:32 to 2:45  What is still wrong, and what is next

**On screen.** The undercut frame, held. Then black with the closing line.

**Narration.**
> Still wrong: a groove at the weld toe, in four frames, reported each time as the
> corroded plate beside it.
>
> Next, a bench plate with hidden wall loss, measured for real.

**Lower third, over the undercut frame.** undercut in frames 1 to 4, never named . it
renders faintly; model or render, not yet separated . next: a bench plate with two
pockets milled into its back face, and a real gauge

---

## Capture checklist

Record these before editing. Everything except the live clip is runnable with no key,
and none of those commands writes a file.

| Clip | Command or file | Length |
| --- | --- | --- |
| The seam shot | `tank/data/shots/seam_shot.mp4` | 10 s |
| Crawler hero still | `tank/data/shots/crawler_hero.png` | still |
| Macro weld still | `tank/data/shots/hero_v2.png` | still |
| Inspection, replayed | `python tank/score_survey.py --replay` | seconds; hold on the summary |
| Probe planning, arithmetic | `python tank/score_probes.py --baselines` | seconds; hold on the table |
| Probe planning, recorded model runs | `python tank/significance.py` | seconds; hold on the Nemotron rows |
| Probe planning, tie-break orders | `python tank/score_probes.py --tiebreaks` | seconds; hold on the median line |
| Live clip, owner's take, paid | `python scripts/demo.py` with a key, service running | about 9 s, captioned as live |
| Rust frame and passed frame | `tank/data/survey/frames/rgb_0005.png` and `rgb_0015.png` | stills |
| Undercut frame | `tank/data/survey/frames/rgb_0003.png` | still |
| Cell result chart | `docs/figures/seam_result.png` | still |

Do not record a tank `--live` run. The one live take is the demo clip above, and only
with the owner's approval. If a tank live run is ever needed for a clip, pass `--out` to a
scratch path outside the repository. Commit the working tree before recording. After
recording, `git diff --exit-code HEAD -- tank/data/survey` must exit 0 and
`git status --porcelain -- tank/data/survey` must print nothing; if either fails,
something wrote over the record and the take is void.

Terminal settings for the recordings: dark background, a font large enough to read at
1080p without zooming, and a window about 110 columns wide so the score table does not
wrap. Clear the screen before each run so nothing from a previous attempt is on screen.

## Rules check

- Under three minutes, on a word count, not yet on a read-through. The narration is 388
  words, segment by segment 43, 55, 30, 51, 82, 40, 56 and 31, which at 150 words a
  minute is about 2:35, plus the 9 second silent live clip: about 2:44. Each segment is
  within half a second of its slot at that rate. Time a read-through at recording pace
  before the take, and cut further if it runs past 2:50.
- 65 seconds of modules in action, which satisfies the Physical AI requirement for a
  project with no hardware component.
- Every number spoken appears on screen, and every number spoken or shown in a lower
  third is printed by a command in the repository:
  23 of 24, 1 missed, 0 false alarms and the undercut in frames 1 to 4 by
  `score_survey.py --replay`; 13 of 13 clean frames passed and 10 of 11 fault frames
  flagged by `significance.py`; 168, 28, 20 and 16, the five hidden patches, the one
  visible patch and two of seven at random by `score_probes.py --baselines`; the median
  of 12 over 5040 tie-break orders by `score_probes.py --tiebreaks`; 9 to 12 and the
  paired p 0.143 to 0.344 by `significance.py`; 88 and 86 of 89, and the three points
  where only Nano is right with its stated reasons, by
  `scripts/eval_router_truth.py --rule`, which pairs the rule with the recorded answers
  and makes no model call (it rewrites `data/router_rule_eval.json` when
  `data/cap50v3` is present; pass `--manifest` an absent path to only read it); the cell
  figures by `scripts/run_seam_campaign.py`, and the 2.82 mm chain by
  `cell/error_budget.py` (its `summary()`, pure arithmetic, set out in the README section
  "What the 0.67 mm does not include"). The confined-space figure is a public BLS figure,
  shown with its source.
- The live clip, if recorded, is captioned as live for its whole length and carries no
  scored number.
- No claim is made about hardware that does not exist.
