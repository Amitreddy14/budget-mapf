# View-radius sweep: results log

Rules are in `docs/PREREG-fov-sweep.md` (unchanged). This file records outcomes.

## a12maze_f3n (fov 3, no distance map): gate FAILED, no verdict

Five seeds per arm, last 4 epochs per run.

| arm | on-goal | success | coll/step | offline err |
|---|---|---|---|---|
| none | 0.0545 +- 0.0057 | 0.0000 | 0.6953 +- 0.1265 | 0.0348 +- 0.0015 |
| shuffled | 0.0330 +- 0.0017 | 0.0000 | 0.6731 +- 0.1877 | 0.0331 +- 0.0011 |
| full | 0.0347 +- 0.0018 | 0.0000 | 0.8062 +- 0.1867 | 0.0325 +- 0.0009 |

The pre-registered gate requires the `none` arm's on-goal in [0.15, 0.85].
It is 0.0545, and no episode was solved in any arm. The setting is
unmeasurable. `headroom_seeds.py` printed "NULL" for full - shuffled
(+0.0016, p = 0.173); that label applies the comparison rule without the gate
and is **not** a finding. No claim is made about communication at fov 3.

What it does show: a cloned agent with a 3x3 window and no distance map cannot
perform the maze task at all, so there is no base behaviour for a message to
improve. Information starvation made the task unlearnable, not
communication-dependent.

## a12maze_f5n (fov 5, no distance map): gate FAILED, no verdict

Five seeds per arm, last 4 epochs per run. 141 demo episodes, 620,472 samples.

| arm | on-goal | success | coll/step | offline err |
|---|---|---|---|---|
| none | 0.0611 +- 0.0027 | 0.0000 | 0.3391 +- 0.0311 | 0.0331 +- 0.0014 |
| shuffled | 0.0406 +- 0.0030 | 0.0000 | 0.4092 +- 0.1155 | 0.0315 +- 0.0010 |
| full | 0.0442 +- 0.0045 | 0.0000 | 0.3189 +- 0.0567 | 0.0313 +- 0.0010 |

`none` on-goal 0.0611 is below the 0.15 gate and nothing was solved. Same
outcome as fov 3. The script printed "NULL" before the gate was added to
`verdict()` (it now prints UNMEASURABLE; tests added). The offline error of
about 0.033 is roughly three times the 0.010 seen with the distance map, in
line with the earlier "errors tripled" reading. Offline accuracy that looks
fine does not translate into closed-loop navigation of a 41x41 maze.

## Next
fov 9 (secondary, alpha = 0.025, same gate) is the one remaining planned
setting. It is not promoted to primary after seeing fov 3 and fov 5.
