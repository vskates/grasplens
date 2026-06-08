from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalBinarySet:
    qhat: float
    alpha: float

    def prediction_sets(self, p_unsafe: np.ndarray) -> np.ndarray:
        """Return columns [safe_in_set, unsafe_in_set]."""

        p_unsafe = np.asarray(p_unsafe, dtype=np.float64)
        p_safe = 1.0 - p_unsafe
        safe_in = (1.0 - p_safe) <= self.qhat
        unsafe_in = (1.0 - p_unsafe) <= self.qhat
        return np.stack([safe_in, unsafe_in], axis=1)

    def accepts(self, p_unsafe: np.ndarray) -> np.ndarray:
        return ~self.prediction_sets(p_unsafe)[:, 1]

    def coverage(self, p_unsafe: np.ndarray, y_unsafe: np.ndarray) -> float:
        sets = self.prediction_sets(p_unsafe)
        y_unsafe = np.asarray(y_unsafe, dtype=np.int64)
        covered = np.where(y_unsafe == 1, sets[:, 1], sets[:, 0])
        return float(np.mean(covered))


def fit_binary_conformal(
    p_unsafe_calib: np.ndarray,
    y_unsafe_calib: np.ndarray,
    alpha: float = 0.1,
) -> ConformalBinarySet:
    p_unsafe_calib = np.asarray(p_unsafe_calib, dtype=np.float64)
    y_unsafe_calib = np.asarray(y_unsafe_calib, dtype=np.int64)
    p_true = np.where(y_unsafe_calib == 1, p_unsafe_calib, 1.0 - p_unsafe_calib)
    scores = 1.0 - p_true
    if len(scores) == 0:
        return ConformalBinarySet(qhat=1.0, alpha=alpha)
    q_level = min(1.0, np.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    qhat = float(np.quantile(scores, q_level, method="higher"))
    return ConformalBinarySet(qhat=qhat, alpha=alpha)

