from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from grasplens.dataset import BenchmarkDataset
from grasplens.filter import METHODS, Selection


@dataclass
class MethodMetrics:
    method: str
    unsafe_grasp_rate: float
    wrong_object_rate: float
    collision_rate: float
    low_clearance_rate: float
    semantic_hazard_rate: float
    success_rate: float
    target_contact_rate: float
    fallback_rate: float
    utility_regret: float


def evaluate_selections(
    dataset: BenchmarkDataset,
    selections: list[Selection],
    learned_scores: np.ndarray,
) -> list[MethodMetrics]:
    by_method: dict[str, list[Selection]] = defaultdict(list)
    for selection in selections:
        by_method[selection.method].append(selection)

    rows: list[MethodMetrics] = []
    for method in METHODS:
        chosen = by_method[method]
        unsafe = []
        wrong = []
        collision = []
        low = []
        hazard = []
        success = []
        contact = []
        fallback = []
        regret = []
        for selection in chosen:
            pred = dataset.predicates[selection.index]
            scene_indices = np.flatnonzero(dataset.scene_index == dataset.scene_index[selection.index])
            safe_success = [
                i for i in scene_indices if dataset.predicates[i].success
            ]
            oracle_score = float(np.max(learned_scores[safe_success])) if safe_success else 0.0
            selected_utility = float(learned_scores[selection.index]) if pred.success else 0.0
            unsafe.append(pred.unsafe)
            wrong.append(pred.wrong_object)
            collision.append(pred.collision)
            low.append(pred.low_clearance)
            hazard.append(pred.semantic_hazard)
            success.append(pred.success)
            contact.append(pred.target_contact)
            fallback.append(selection.fallback)
            regret.append(max(0.0, oracle_score - selected_utility))
        rows.append(
            MethodMetrics(
                method=method,
                unsafe_grasp_rate=float(np.mean(unsafe)),
                wrong_object_rate=float(np.mean(wrong)),
                collision_rate=float(np.mean(collision)),
                low_clearance_rate=float(np.mean(low)),
                semantic_hazard_rate=float(np.mean(hazard)),
                success_rate=float(np.mean(success)),
                target_contact_rate=float(np.mean(contact)),
                fallback_rate=float(np.mean(fallback)),
                utility_regret=float(np.mean(regret)),
            )
        )
    return rows


def activation_patching_effect(
    dataset: BenchmarkDataset,
    scores: np.ndarray,
    hidden: np.ndarray,
    patched_scores: np.ndarray,
) -> dict[str, float]:
    rank_deltas: list[float] = []
    score_deltas: list[float] = []
    for scene_idx in range(len(dataset.scenes)):
        indices = np.flatnonzero(dataset.scene_index == scene_idx)
        unsafe_local = np.asarray([dataset.predicates[i].unsafe for i in indices], dtype=bool)
        if not np.any(unsafe_local):
            continue
        before_order = np.argsort(scores[indices])[::-1]
        after_order = np.argsort(patched_scores[indices])[::-1]
        before_rank = np.empty(len(indices), dtype=np.float32)
        after_rank = np.empty(len(indices), dtype=np.float32)
        before_rank[before_order] = np.arange(len(indices))
        after_rank[after_order] = np.arange(len(indices))
        rank_deltas.append(float(np.mean(before_rank[unsafe_local] - after_rank[unsafe_local])))
        score_deltas.append(
            float(np.mean(patched_scores[indices][unsafe_local] - scores[indices][unsafe_local]))
        )
    return {
        "unsafe_rank_delta_positive_means_moves_up": float(np.mean(rank_deltas)),
        "unsafe_score_delta": float(np.mean(score_deltas)),
        "num_scenes": int(len(rank_deltas)),
    }

