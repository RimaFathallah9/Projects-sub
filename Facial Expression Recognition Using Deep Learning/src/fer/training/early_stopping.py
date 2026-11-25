from __future__ import annotations


class EarlyStopping:
    """Stops when the monitored value has not improved by > min_delta for `patience` epochs."""

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 0.0):
        assert mode in {"min", "max"}
        self.patience, self.mode, self.min_delta = patience, mode, min_delta
        self.best: float | None = None
        self.best_epoch = 0
        self.counter = 0

    def is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        return value > self.best + self.min_delta if self.mode == "max" else value < self.best - self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Returns True if `value` is a new best. Check `.should_stop` afterwards."""
        if self.is_better(value):
            self.best, self.best_epoch, self.counter = value, epoch, 0
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.counter >= self.patience
