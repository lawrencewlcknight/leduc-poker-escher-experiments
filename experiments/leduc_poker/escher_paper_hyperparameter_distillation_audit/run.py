"""Run Experiment 46 using the validated Experiment 45 implementation."""

from __future__ import annotations

from contextlib import contextmanager
import sys
from typing import Iterable

from experiments.leduc_poker.escher_frozen_policy_distillation_audit import (
    run as _base,
)

from .config import (
    ARMS,
    ARM_ORDER,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    validate_config,
)


_CONTRACT = {
    "ARMS": ARMS,
    "ARM_ORDER": ARM_ORDER,
    "DEFAULT_CONFIG": DEFAULT_CONFIG,
    "DEFAULT_SEEDS": DEFAULT_SEEDS,
    "EXPERIMENT_ID": EXPERIMENT_ID,
    "EXPERIMENT_NAME": EXPERIMENT_NAME,
    "validate_config": validate_config,
}


@contextmanager
def experiment_contract():
    """Temporarily bind the shared runner to Experiment 46's frozen contract."""
    previous = {name: getattr(_base, name) for name in _CONTRACT}
    try:
        for name, value in _CONTRACT.items():
            setattr(_base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(_base, name, value)


def build_config(args):
    with experiment_contract():
        return _base.build_config(args)


def _aggregate(*args, **kwargs):
    with experiment_contract():
        return _base._aggregate(*args, **kwargs)  # pylint: disable=protected-access


def main(argv: Iterable[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(
        item == "--output-root" or item.startswith("--output-root=")
        for item in arguments
    ):
        arguments.extend([
            "--output-root",
            "outputs/paper_hyperparameter_distillation_audit",
        ])
    with experiment_contract():
        return _base.main(arguments)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

