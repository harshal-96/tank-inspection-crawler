# Retired four point probe runs

Probe planner runs on the earlier survey scene, which had one hidden patch and
4 rejectable probe points out of 168 (each file records `"rejectable": 4`).
Two planners compared on four points cannot be told apart, so the scene was
given five hidden patches and 28 rejectable points (commit 24208a0), and these
results were retired. They are kept as the record of that run and are quoted
nowhere as a current result.

- `probing_run3.json`, `probing_run4.json`, `probing_run5.json`: the arithmetic
  baselines, Nemotron 3 Nano and Nemotron 3 Super, one run each.
- `probing_floor.json`: Nemotron 3 Nano with the rule floor
  (`probe_planner.plan_with_floor`), five runs.

The current results are the `probing*.json` files one level up, all on 28
rejectable points.
