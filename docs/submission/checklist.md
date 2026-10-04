# Submission checklist

Against the official rules at https://nebiusglobalaihackathon.devpost.com/rules, read
on 23 September 2026. Deadline **30 October 2026, 10:00 Pacific**.

## Hard requirements

| | State | Note |
| --- | --- | --- |
| Runs on Nebius Token Factory or AI Cloud | **done** | runtime calls to the Token Factory inference API, which is the rule's own definition |
| Uses at least one NVIDIA open source model | **done** | Nemotron Nano plans the tank's probing, Nano does the cell's view planning, Lightning runs the supervisor and writes the briefs |
| Open source licence file in the repo | **done** | Apache 2.0 at the root, explicitly named as acceptable |
| README with setup and run instructions | **done** | verified working: service starts, `scripts/demo.py --offline` populates the cell in 2.8 s |
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

**If and when it is published, the history is clean.** All 115 commits were scanned for
files that were ever added and later gitignored, and for key-shaped strings. Nothing
sensitive has ever been committed: `.env` has been ignored from the start and only
`.env.example` is tracked. That matters because gitignoring a file today does nothing
about a file committed last month, and publishing would expose the whole history, not
the working tree. Re-check before pushing with:

```
git log --all --pretty=format: --name-only --diff-filter=A | sort -u | grep -iE "\.env$|secret|\.key$"
```

Three ways to resolve it are set out at the end of `devpost.md` discussion notes:
publish and rely on the Apache licence, publish the tank half only, or do not submit
and use the work for customer conversations instead. It is a decision, not an oversight.

## Before recording the video

- [ ] Finish the 240 frame seam shot and encode it: `python scripts/encode_shot.py`
- [ ] Screen record `python tank/score_survey.py` at full speed
- [ ] Screen record `python tank/score_probes.py --repeat 3`
- [ ] Check every number spoken in the script against the current README

## Numbers as they stand

Each is reproducible from the repository with the command beside it.

| | | |
| --- | --- | --- |
| Tank, pass or defect | 24 of 24, nothing missed, no false alarms, none declined | `tank/score_survey.py` |
| Tank, naming the worst fault | 20 of 24 | same |
| Tank, every fault found | 9 of 13, none invented | same |
| Probe planning, arithmetic sweep | **16 of 28**, every run | `tank/score_probes.py --repeat 3` |
| Probe planning, Nemotron Nano | 9 to 12 of 28 | same |
| Probe planning, follow the camera | 0 of 28 | same |
| What those counts license | confidence bounds and paired test | `tank/significance.py` |
| Cell, seam tracking | 22.802 mm to 0.668 mm, 50 of 50 under a millimetre | `scripts/run_seam_campaign.py` |
| Cell, routing | 88 of 89 | `scripts/eval_router_truth.py` |
| Cell, bead inspection | 91% pass or defect on 150 beads, no false alarms | `scripts/run_inspection.py` |
| Supervisor | 9 of 10 | `scripts/eval_supervisor.py` |
| Briefs | 6 of 6 on all four checks | `scripts/eval_briefs.py --limit 6` |
| Test suite | 292 passing | `pytest -q` |

## Known weaknesses, stated rather than hidden

- **Undercut.** The inspector reports the corroded plate beside a weld-toe groove rather
  than the groove. Every image-capable model on the account does the same, so it is the
  simulated groove at fault. Written up in the README.
- **The model planner loses to arithmetic**, 9 to 12 of 28 against a deterministic
  16. Reported rather than buried: sparse defects on a fixed budget reward coverage
  over reasoning, and the model belongs on the picture instead.
- **No hardware.** Nothing physical exists yet. The video uses the no-hardware branch of
  the rule and makes no claim otherwise.
