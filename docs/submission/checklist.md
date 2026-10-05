# Submission checklist

Against the official rules at https://nebiusglobalaihackathon.devpost.com/rules, read
on 23 September 2026. Deadline **30 October 2026, 10:00 Pacific**.

## Hard requirements

| | State | Note |
| --- | --- | --- |
| Runs on Nebius Token Factory or AI Cloud | **done** | runtime calls to the Token Factory inference API, which is the rule's own definition |
| Uses at least one NVIDIA open source model | **done** | live calls in the welding cell: Nemotron 3 Nano makes every per-part view decision, Nemotron 3.5 Lightning runs the supervisor and writes the corrective action briefs. In the tank, Nemotron 3 Nano was measured as the probe planner and is reported as a negative result; the coverage rule is what runs |
| Open source licence file in the repo | **done** | Apache 2.0 at the root, explicitly named as acceptable |
| README with setup and run instructions | **done** | verified working: service starts, `scripts/demo.py --offline` populates the cell in about three seconds |
| Public repo on GitHub, GitLab or Bitbucket | **NOT DONE** | held at the owner's request. This is the one item that voids the entry |
| Video under 3 minutes | **not done** | script and shot list written, footage partly rendered |
| At least 1 minute of hardware operating, or key modules in action | **ready** | script allocates 65 seconds to modules running, which satisfies the no-hardware branch of the rule |
| Demo URL | **not required** | Physical AI is the one track exempt from this |
| Text description | **done** | `docs/submission/devpost.md` |
| One track selected | Physical AI | |
| Feedback on tools used | **done** | `docs/submission/tools_feedback.md` |
| Project new or significantly updated in the window | **done** | all work falls inside 26 Aug to 30 Oct 2026 |
| Eligibility | **ok** | India is not on the excluded list |

## The one blocker

The repository is still local, with no git remote configured. A public repo is a hard
requirement and there is no private submission route. Nothing else on this list can
compensate for it.

**If and when it is published, the history should be clean.** The first 115 commits
were scanned for files that were ever added and later gitignored, and for key-shaped
strings, and nothing sensitive turned up: `.env` has been ignored from the start and
only `.env.example` is tracked. On 5 October 2026 the filename check below was re-run
over all 128 commits and matched nothing; the key-shaped string scan has not been
repeated over the later commits and should be before pushing. That matters because
gitignoring a file today does nothing about a file committed last month, and publishing
would expose the whole history, not the working tree.

```
git log --all --pretty=format: --name-only --diff-filter=A | sort -u | grep -iE "\.env$|secret|\.key$"
```

## Before recording the video

- [ ] Finish the 240 frame seam shot and encode it: `python scripts/encode_shot.py`
- [ ] Screen record `python tank/score_survey.py --replay`, captioned as a replay of
      recorded verdicts
- [ ] Screen record `python tank/score_probes.py --baselines` and
      `python tank/significance.py`, captioned the same way
- [ ] Do not record a `--live` run; if one is ever needed, pass `--out` to a scratch path
      outside the repository
- [ ] Commit the working tree before recording
- [ ] After recording, `git diff --exit-code HEAD -- tank/data/survey` exits 0 and
      `git status --porcelain -- tank/data/survey` prints nothing
- [ ] Check every number spoken in the script against the current README

## Numbers as they stand

Each is printed by the command beside it. None of the tank commands calls a model or
writes a file, with or without a key set. The inspection figures are DeepSeek V4.1
Flash's recorded verdicts on the 24-frame development set.

| | | |
| --- | --- | --- |
| Tank, pass or defect | 23 of 24; 1 missed (station 12, a crack frame 11 had named), 0 false alarms, 0 declined | `python tank/score_survey.py --replay` |
| Tank, the same call only when the named fault is in view | 21 of 24 | same |
| Tank, naming the worst fault | 19 of 24 | same |
| Tank, every fault found | 9 of 14, none invented; every fault found in 19 of 24 frames | same |
| Tank, distinct faults named at least once | 4 of 5 (never: D001 undercut, frames 1 to 4) | same |
| Tank, one sided 95% exact lower bounds | 0.817, 0.708, 0.611, 0.390, 0.343 | `python tank/significance.py` |
| Probe, the task | 28 of 168 rejectable, 20 of them at stations the inspector passed, on 5 hidden patches | `python tank/score_probes.py --baselines` |
| Probe, coverage rule as shipped | 16 of 28, 5 of 5 patches | same |
| Probe, the rule given only the model's view | 11 of 28, 5 of 5 patches | same |
| Probe, random two of seven | 8 expected, central 95% 4 to 12 | same |
| Probe, on the seam / beside reported rust | 2 of 28 each, 1 of 5 patches | same |
| Probe, oracle | 23 of 28 | same |
| Probe, the rule under all 5040 tie-break orders | 7 to 18, median 12; 93.9% of orders find fewer than the shipped one | `python tank/score_probes.py --tiebreaks` |
| Probe, the rule on 1000 random layouts | shipped layout better for it than 94.9% of layouts; median 40.7% found | `python tank/layout_sweep.py` |
| Probe, Nemotron 3 Nano, recorded runs | 9, 10 and 12 of 28, 4 of 5 patches each; camera hidden 11 | `python tank/significance.py` |
| Probe, exact McNemar per run | p 0.143, 0.344, 0.146; camera hidden 0.267 | same |
| A real 473 m tank | 1.3 h driving, 7.0 h vision serial; a model planner adds 5.3 h (4.0 overlapped) | same |
| Crawler adhesion | holds safety factor 3 dry at 0.2 mm liftoff on the assumed 220 N a wheel; about 435 N needed in the worst dry case | `python tank/adhesion.py` |
| Cell, seam tracking | 22.802 mm to 0.668 mm, 50 of 50 under a millimetre (arrival spread exaggerated for visibility); tracker on exact rendered depth, whole chain 2.82 mm against a 1.5 mm requirement | `scripts/run_seam_campaign.py`; the chain by `cell/error_budget.py` |
| Cell, per-part view decision | 88 of 89; the one miss gave up 0.30 mm | `scripts/eval_router_truth.py` (recorded: `data/router_truth_eval.log`) |
| Cell, bead inspection | 91% pass or defect on 150 beads, no false alarms | `scripts/run_inspection.py` |
| Supervisor | 9 of 10 | `scripts/eval_supervisor.py` |
| Briefs | 6 of 6 on all four checks | `scripts/eval_briefs.py --limit 6` |
| Test suite | see the count printed by the command; recount before quoting | `python -m pytest -q` |

The cell scripts are not covered by the no-model promise above. Read each one's
`--help` before running it with a key set.

## Known weaknesses, stated rather than hidden

- **Undercut.** Four frames carry a groove at the weld toe and the inspector reports the
  corroded plate beside it every time. That is the recorded fact; whether the cause is
  the model or the rendered groove has not been separated. Written up in the README.
- **One missed crack.** Station 12 holds the same crack frame 11 named, and it was
  passed. An earlier answer key hid it, and that key was corrected.
- **The model planner was never ahead of arithmetic**, 9 to 12 of 28 against the rule's
  16, but on matched information the rule finds 11, level with the model. The rule
  drives the arm because it is no worse, deterministic, free and adds no hours, not
  because coverage was shown to reason better.
- **Development set, one pass.** The 24 frames were corrected after reading model
  answers on them, and each station was inspected once.
- **No hardware.** Nothing physical exists yet. The video uses the no-hardware branch of
  the rule and makes no claim otherwise.
