"""Cloud adapter for Experiment 48."""

from __future__ import annotations

from contextlib import contextmanager

from experiments.leduc_poker.escher_frozen_policy_distillation_audit import (
    cloud as _base,
)
from . import run as experiment_run
from .config import ARM_ORDER, DEFAULT_SEEDS, EXPERIMENT_ID, EXPERIMENT_NAME


task_name = _base.task_name


@contextmanager
def experiment_contract():
    replacements = {
        "ARM_ORDER": ARM_ORDER,
        "DEFAULT_SEEDS": DEFAULT_SEEDS,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "EXPERIMENT_NAME": EXPERIMENT_NAME,
        "run_experiment": experiment_run.main,
        "_aggregate": experiment_run._aggregate,  # pylint: disable=protected-access
    }
    previous = {name: getattr(_base, name) for name in replacements}
    with experiment_run.experiment_contract():
        try:
            for name, value in replacements.items():
                setattr(_base, name, value)
            yield
        finally:
            for name, value in previous.items():
                setattr(_base, name, value)


def main() -> None:
    with experiment_contract():
        _base.main()


if __name__ == "__main__":  # pragma: no cover
    main()
