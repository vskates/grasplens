from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grasplens.conformal import ConformalBinarySet
from grasplens.dataset import BenchmarkDataset


METHODS = ("baseline", "geometry", "probe", "grasplens_full")


@dataclass(frozen=True)
class Selection:
    method: str
    index: int
    fallback: bool = False


def _choose_best(indices: np.ndarray, scores: np.ndarray) -> int:
    return int(indices[np.argmax(scores[indices])])


def _soft_fallback(indices: np.ndarray, scores: np.ndarray, risk: np.ndarray) -> int:
    utility = scores[indices] - 1.2 * risk[indices]
    return int(indices[np.argmax(utility)])


def select_for_scene(
    dataset: BenchmarkDataset,
    scene_idx: int,
    scores: np.ndarray,
    unsafe_prob: np.ndarray,
    conformal: ConformalBinarySet,
    risk_threshold: float = 0.42,
) -> list[Selection]:
    indices = np.flatnonzero(dataset.scene_index == scene_idx)
    pred = dataset.predicates
    baseline = _choose_best(indices, scores)

    hard_safe = np.asarray([pred[i].hard_safe for i in indices], dtype=bool)
    geometry_pool = indices[hard_safe]
    if len(geometry_pool) > 0:
        geometry = _choose_best(geometry_pool, scores)
        geometry_fallback = False
    else:
        geometry = _soft_fallback(indices, scores, unsafe_prob)
        geometry_fallback = True

    conformal_accept = conformal.accepts(unsafe_prob[indices])
    probe_accept = conformal_accept & (unsafe_prob[indices] <= risk_threshold)
    probe_pool = indices[probe_accept]
    if len(probe_pool) > 0:
        probe = _choose_best(probe_pool, scores)
        probe_fallback = False
    else:
        probe = _soft_fallback(indices, scores, unsafe_prob)
        probe_fallback = True

    full_accept = hard_safe & probe_accept
    full_pool = indices[full_accept]
    if len(full_pool) > 0:
        full = _choose_best(full_pool, scores)
        full_fallback = False
    elif len(geometry_pool) > 0:
        full = _soft_fallback(geometry_pool, scores, unsafe_prob)
        full_fallback = True
    else:
        full = _soft_fallback(indices, scores, unsafe_prob)
        full_fallback = True

    return [
        Selection("baseline", baseline),
        Selection("geometry", geometry, geometry_fallback),
        Selection("probe", probe, probe_fallback),
        Selection("grasplens_full", full, full_fallback),
    ]


def select_all(
    dataset: BenchmarkDataset,
    scores: np.ndarray,
    unsafe_prob: np.ndarray,
    conformal: ConformalBinarySet,
    risk_threshold: float = 0.42,
) -> list[Selection]:
    selections: list[Selection] = []
    for scene_idx in range(len(dataset.scenes)):
        selections.extend(
            select_for_scene(dataset, scene_idx, scores, unsafe_prob, conformal, risk_threshold)
        )
    return selections

