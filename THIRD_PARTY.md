# Third-party components

## EV-UAV dataset and official evaluator

EventRefine reads the public EV-UAV archive format (`train/*.npz`, `val/*.npz`, and, only for an explicitly authorized final evaluation, `test/*.npz`). The dataset and the official evaluator are external inputs; they are not redistributed in this repository.

The upstream EV-UAV repository used for format reference is:

- https://github.com/ChenYichen9527/Ev-UAV

The inspected upstream snapshot did not contain a clearly identified `LICENSE` or `COPYING` file. EventRefine therefore does not vendor upstream implementation code. If you use the official evaluator, provide its checkout with `--official-evaluator-root` and follow its own licensing terms.

## PyTorch, NumPy, PyYAML

These are runtime dependencies installed from their respective upstream package distributions under their own licenses.
