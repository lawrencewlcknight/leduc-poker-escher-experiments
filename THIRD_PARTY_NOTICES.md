# Third-party notices

## Sandholm-Lab ESCHER parallel architecture

`escher_poker/parallel_solver.py` adapts the learner/experience-worker
architecture from
[`parallelized_ESCHER.py`](https://github.com/Sandholm-Lab/ESCHER/blob/e694eaaa251952696aaf36ef1c790887c8324750/parallelized_ESCHER.py)
at upstream commit `e694eaaa251952696aaf36ef1c790887c8324750`.

The upstream file states:

- Original Deep CFR code copyright 2019 DeepMind Technologies Limited.
- ESCHER code copyright 2022 Stephen McAleer.
- Licensed under the Apache License, Version 2.0.

The adaptation uses this repository's current solver, networks, replay
backends, target processing, diagnostics, and experiment infrastructure. It
also partitions the global traversal and replay budgets across workers, rather
than multiplying those budgets by the number of workers.
