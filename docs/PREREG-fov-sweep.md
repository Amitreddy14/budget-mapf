# Pre-registration: view-radius sweep without a distance map

Written 2026-11-25, **before any run**. Edits after the first run must be
appended below the line "Amendments", not made in place.

## Question
Does correct message addressing help once the agents are made information-
starved? Earlier readings (section 21 and the main study) used a 9x9 window
plus a globally computed distance map, which leaves little for a message to
carry.

## Settings (12 agents, 41x41 maze, 768 steps, peer goals hidden)
| tag | fov | distance map |
|---|---|---|
| a12maze_f3n | 3 | none |
| a12maze_f5n | 5 | none |
| a12maze_f9n | 9 | none |

Communication radius stays at 8 in every setting, so a message can cover more
than the agent's own view. Arms, epochs, window, seeds and the Welch test are
exactly as in `scripts/headroom_seeds.py` (section 21).

## Gate (applies before any comparison is read)
A setting is **measurable** only if the `none` arm's mean on-goal, over 5
seeds, lies in [0.15, 0.85]. Below 0.15 the agents are at the floor and a
message gap cannot show; above 0.85 there is no headroom. An unmeasurable
setting is reported as such, with no verdict.

## Primary test (one test)
Setting a12maze_f3n, `full - shuffled` on closed-loop on-goal, last 4 epochs
per run, Welch across 5 seeds, alpha = 0.05, with offline error agreeing in
sign. Verdict strings are those of `headroom_seeds.py`.

## Secondary tests
fov 5 and fov 9 under the same rule, alpha = 0.025 each (Bonferroni over the
two). Reported whether or not the primary is significant.

## Procedure
Seed 0 of all three arms is run first at fov 3 and used **only** for the gate
and to confirm nothing crashes. No full-versus-shuffled conclusion is drawn
from it. Seeds 1-4 follow regardless.

## What each outcome means
- Primary SIGNAL: communication helps when the view is small. Report the
  boundary from the secondaries.
- Primary NULL with a measurable gate: the null holds in the setting most
  favourable to messages. Report the CI, not "no effect".
- Gate failure: the setting is uninformative; say so.

## Amendments
(none)
