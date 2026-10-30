# Submission checklist

Against the official rules at https://nebiusglobalaihackathon.devpost.com/rules, read
on 23 September 2026. Deadline **30 October 2026, 10:00 Pacific**.

## Hard requirements

| | State | Note |
| --- | --- | --- |
| Runs on Nebius Token Factory or AI Cloud | **done** | runtime calls to the Token Factory inference API, which is the rule's own definition |
| Uses at least one NVIDIA open source model | **done** | live calls in the welding cell: with a key, `scripts/demo.py` runs `scripts/run_campaign.py --routed`, where Nemotron 3 Nano makes every per-part view decision (88 of 89 against ground truth; the prompt's limits as arithmetic get 86, exact paired p 0.625, and at the three points only Nano gets right its stated reason misreads the inlier balance), and Nemotron 3.5 Lightning runs the supervisor and writes the corrective action briefs. In the tank no NVIDIA model runs: Nemotron 3 Nano was measured as the probe planner and is reported as a negative result; the coverage rule is what runs |
| Open source licence file in the repo | **done** | Apache 2.0 at the root, explicitly named as acceptable |
| README with setup and run instructions | **done** | verified working: service starts, `scripts/demo.py --offline` populates the cell in about three seconds |
| Public repo on GitHub, GitLab or Bitbucket | **NOT DONE** | owner's step, below. This is the one item that voids the entry |
| Video under 3 minutes | **rendered, not uploaded** | 2:58, narrated and subtitled, from recorded runs only (`docs/submission/video_script.md`). Uploading it and putting its link in the submission is the owner's step, below |
| At least 1 minute of hardware operating, or key modules in action | **done** | 77 seconds of the modules running (chapters 03 to 07), plus 20 seconds of the welding cell, which satisfies the no-hardware branch of the rule. No hardware is claimed |
| Demo URL | **not required** | Physical AI is the one track exempt from this |
| Text description | **done** | `docs/submission/devpost.md`, once the repository URL is filled in |
| One track selected | Physical AI | |
| Feedback on tools used | **done** | `docs/submission/tools_feedback.md` |
| Project new or significantly updated in the window | **done** | all work falls inside 26 Aug to 30 Oct 2026 |
| Eligibility | **ok** | India is not on the excluded list |

## Owner's steps before the deadline

These are the owner's to do. None of them has been done for you, and the first one
decides whether the entry counts at all.

1. **Commit the working tree.** Everything below is checked against committed history,
   and publishing exposes the history, not the working tree.
2. **Repeat the history scan over every commit, after the last one is made.** On
   5 October 2026 the history held 130 commits, no key-shaped string, and `.env` had
   never been committed; only `.env.example` is tracked. Re-run the same day at commit
   eaeba90, over 131 commits, it matched nothing again: the configured key values were
   found in 0 commits, no `sk-`, `tvly-` or JWT-shaped string and no long value assigned
   to an API key or token turned up in any patch, and `.env` was never added. Commits made
   after that are not covered, so repeat it before pushing. That matters because
   gitignoring a file today does nothing about a file committed last month. The filename
   check:

   ```
   git log --all --pretty=format: --name-only --diff-filter=A | sort -u | grep -iE "\.env$|secret|\.key$"
   ```

   prints nothing when the history is clean. For the strings, search every patch
   (`git log --all -p`) for the key values themselves and for key-shaped strings, and
   print counts, never the matches.
3. **Create the public repository** on GitHub, GitLab or Bitbucket and push to it. There
   is no git remote configured yet, and there is no private submission route.
4. **Make the submission the branch a visitor lands on.** `main` is 105 commits behind
   `self-correcting-cell` and still shows the September welding-cell README. `main` is an
   ancestor of the branch, so either make `self-correcting-cell` the default branch, or
   fast-forward: `git checkout main && git merge --ff-only self-correcting-cell`, then
   push it: `git push origin main self-correcting-cell`. Then open the remote's default
   branch page and check that it shows the tank README, not the September one.
5. **Put the real URL in.** Replace `REPOSITORY_URL` in the "Try it" section of
   `docs/submission/devpost.md`, and `git clone <this-repo>` in the README's Setup
   section, with the real URL and folder name. Then run the README's ten-minute block
   from a fresh clone of the remote.
6. **Upload the video** to YouTube as **public**, which the rules ask for ("made publicly
   visible on YouTube"), and paste its link into the
   submission form. The description is ready to paste; it is kept next to the film,
   outside the repository. Put the repository URL in it. The
   film shows no live call, so none needs recording; every number in it comes from a
   replay.
7. **Quote pytest's own count.** Before quoting a test count anywhere, run
   `python -m pytest -q` on the committed tree and use the count it prints, not a number
   from this file or the README.

## The video

- [x] Rendered from the 240 frame seam shot, the 24 survey frames, the recorded verdicts,
      the manifest's probe grid and the dashboard, with no model call
- [x] Opens on the US Bureau of Labor Statistics confined-space figures, then the
      illustrated scaffold fall (tagged ILLUSTRATION); the crawler first appears after it
- [x] Every number spoken or shown checked against the repository (the rules check at the
      end of `docs/submission/video_script.md`)
- [x] `tank/data/survey` untouched: the film reads it and writes nothing into the
      repository
- [ ] Watch it through once, with sound, before uploading

## Numbers as they stand

Each is printed by the command beside it. None of the tank commands calls a model or
writes a file, with or without a key set. The inspection figures are DeepSeek V4.1
Flash's recorded verdicts on the 24-frame development set.

| | | |
| --- | --- | --- |
| Tank, pass or defect | 23 of 24; 1 missed (station 12, a crack frame 11 had named), 0 false alarms, 0 declined; calling every frame sound scores 13 of 24 | `python tank/score_survey.py --replay` |
| Tank, the same call only when the named fault is in view | 22 of 24 | same |
| Tank, naming the worst fault | 19 of 24 | same |
| Tank, every fault found | 10 of 15, none invented; every fault found in 19 of 24 frames | same |
| Tank, distinct faults named at least once | 4 of 5 (never: D001 undercut, frames 1 to 4) | same |
| Tank, one sided 95% exact lower bounds | 0.817, 0.760, 0.611, 0.423, 0.611 (every fault found in a frame), 0.343 | `python tank/significance.py` |
| Tank, by class | frames holding a fault flagged 10 of 11 (0.636), flagged for a fault really in view 9 of 11 (0.530); clean frames passed 13 of 13 (0.794) | same |
| Probe, the task | 28 of 168 rejectable, 20 of them under frames the inspector passed, 18 under frames with no visible fault, on 5 hidden patches | `python tank/score_probes.py --baselines` |
| Probe, coverage rule as shipped | 16 of 28, 5 of 5 patches | same |
| Probe, the rule given only the model's view | 11 of 28, 5 of 5 patches | same |
| Probe, random two of seven | 8 expected, central 95% 4 to 12 | same |
| Probe, on the seam / beside reported rust | 2 of 28 each, 1 of 5 patches | same |
| Probe, oracle | 23 of 28 | same |
| Probe, the rule under all 5040 tie-break orders | 7 to 18, median 12; 93.9% of orders find fewer than the shipped one. Shown only the model's view, 7 to 19, median 13 | `python tank/score_probes.py --tiebreaks` |
| Probe, the rule on 1000 random layouts | shipped layout better for it than 94.9% of layouts; median 40.7% found | `python tank/layout_sweep.py` |
| Probe, Nemotron 3 Nano, recorded runs | 9, 12 and 10 of 28, 4 of 5 patches each; camera hidden 11 (one run). In the run that found 10, station 10 fell back to the rule, so the model's own picks found 9. Replies that named fewer than two offsets were also topped up from the rule, and these runs did not record where, so every model count may include some rule picks | `python tank/significance.py` |
| Probe, exact McNemar per run against the shipped rule | p 0.143, 0.344, 0.146; camera hidden 0.267. With station 10 left out of both arms, still 0.146 | same |
| Probe, the archived four-point scene | p 0.5, 1.0 and 0.5 for the three Nano runs; on 4 points the smallest attainable two-sided p is 0.125 | same |
| A real 473 m tank | 1.3 h driving, 7.0 h vision serial at the median of 9.2 s a frame (7.7 h at the mean of 10.2 s); a model planner adds 5.3 h (4.0 overlapped) | same |
| Crawler adhesion | holds safety factor 3 dry at 0.2 mm liftoff on the assumed 220 N a wheel; about 435 N needed in the worst dry case | `python tank/adhesion.py` |
| Cell, seam tracking | 22.802 mm to 0.668 mm, 50 of 50 under a millimetre (arrival spread exaggerated for visibility; on a real line the joint moves with part variation); tracker on exact rendered depth, whole chain 2.82 mm against a 1.5 mm requirement | `scripts/run_seam_campaign.py`; the chain by `cell/error_budget.py` |
| Cell, per-part view decision | 88 of 89 for Nano, Lightning and Ultra, 61 for Super; the one Nano miss gave up 0.30 mm | `python scripts/eval_router_truth.py --rule` prints the recorded counts per arm, no model call (recorded: `data/router_truth_eval.jsonl`, 89 rows per arm). Do not run it without `--rule` with a key set: that calls a model on every point |
| Cell, the same limits as arithmetic | 86 of 89 against Nano's 88; 4 disagreements, exact McNemar p 0.625. At all three points where Nano alone is right, its stated reason misreads the inlier balance as at least 0.60 when it is 0.51 to 0.59, so it was right there for a reason the evidence contradicts | `python scripts/eval_router_truth.py --rule`, no model call (recorded: `data/router_rule_eval.json`). It rewrites `data/router_rule_eval.json` when `data/cap50v3` is present; pass `--manifest` an absent path to only read it |
| Cell, bead inspection | 91% pass or defect on 150 beads, no false alarms | `scripts/run_inspection.py` |
| Supervisor | 9 of 10 | `scripts/eval_supervisor.py` |
| Briefs | 6 of 6 on all four checks | `scripts/eval_briefs.py --limit 6` |
| Test suite | quote only the count printed by the command, on the committed tree | `python -m pytest -q` |

The cell scripts other than `eval_router_truth.py --rule` are not covered by the
no-model promise above. Read each one's `--help` before running it with a key set.

## Known weaknesses, stated rather than hidden

- **Undercut.** Four frames carry a groove at the weld toe and the inspector reports the
  corroded plate beside it every time. In those frames the groove renders faintly, a thin
  line rather than the dark groove the prompt describes; whether the cause is the model
  or the render has not been separated. Written up in the README.
- **One missed crack.** Station 12 holds the same crack frame 11 named, and it was
  passed. An earlier answer key hid it, and that key was corrected.
- **The model planner was never ahead of the rule at its shipped tie-break order** in a
  recorded run, 9 to 12 of 28 against its 16. That 16 is a lucky order (median
  12 over all 5040), on a layout placed by hand in coverage's favour, and shown only what
  the model sees the rule is no worse (median 13). No comparison is significant. There
  is no evidence the model places probes better, and only weak evidence it places them
  worse. The rule drives the arm because it is no worse, deterministic, free and adds no
  hours.
- **The view decision does not beat its own limits.** Nano's 88 of 89 against the rule's
  86 is four disagreements, p 0.625, and at all three points where Nano alone is right
  its stated reason misreads the inlier balance as at least 0.60 when it is 0.51 to
  0.59, so it was right there for a reason the evidence contradicts. The planner's
  arithmetic fallback used to read the signals under different keys from the ones the
  cell passes, and asked for another look on all 89 points; it is fixed, and
  `tests/test_router_rule.py` pins it.
- **Development set, one pass.** The 24 frames were corrected after reading model
  answers on them, and each station was inspected once.
- **No hardware.** Nothing physical exists yet. The video uses the no-hardware branch of
  the rule and makes no claim otherwise.
