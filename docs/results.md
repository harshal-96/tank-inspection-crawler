# Measured results

Joint position error against the true joint, in millimetres. Parts arrive
under physics with randomised offset. Every arm reads the same renders of the
same parts, and turning the part about the joint axis leaves the joint itself
where it is, so the target never moves between views.

Parts per campaign: 50.

| | Measured | Mean | Median | p95 | Worst | Sub-millimetre |
| --- | --- | --- | --- | --- | --- | --- |
| No correction | 50 of 50 | 22.41 | 21.94 | 36.11 | 50.17 | 0 of 50 |
| One fixed view, 50 parts, guided loop | 50 of 50 | 0.91 | 0.95 | 1.61 | 1.79 | 26 of 50 |
| Look, judge, look again, starting 50 parts, guided loop | 50 of 50 | 0.89 | 0.92 | 1.61 | 1.79 | 27 of 50 |

## What the loop costs

- Starting 50 parts, guided loop: 1.78 looks per part on average, at most 4. 89 model calls over 50 parts, median 2117 ms each.
