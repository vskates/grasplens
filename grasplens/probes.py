from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    concept: str
    model: LogisticRegression
    scaler: StandardScaler
    auroc: float
    average_precision: float
    sparsity: int
    top_dims: list[int]
    top_weights: list[float]

    def predict_proba(self, hidden: np.ndarray) -> np.ndarray:
        hz = self.scaler.transform(hidden)
        return self.model.predict_proba(hz)[:, 1]

    @property
    def direction(self) -> np.ndarray:
        weights = self.model.coef_[0] / (self.scaler.scale_ + 1e-8)
        norm = np.linalg.norm(weights) + 1e-8
        return (weights / norm).astype(np.float32)


def _safe_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def _safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def train_sparse_probe(
    concept: str,
    train_hidden: np.ndarray,
    train_labels: np.ndarray,
    eval_hidden: np.ndarray,
    eval_labels: np.ndarray,
    c: float = 0.18,
) -> ProbeResult:
    scaler = StandardScaler()
    hz_train = scaler.fit_transform(train_hidden)
    hz_eval = scaler.transform(eval_hidden)
    if len(np.unique(train_labels)) < 2:
        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        model.coef_ = np.zeros((1, train_hidden.shape[1]))
        model.intercept_ = np.array([-20.0 if train_labels.mean() < 0.5 else 20.0])
        score = np.full(len(eval_labels), float(train_labels.mean()))
    else:
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=c,
            class_weight="balanced",
            max_iter=500,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="'penalty' was deprecated.*")
            warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1.*")
            model.fit(hz_train, train_labels)
        score = model.predict_proba(hz_eval)[:, 1]
    weights = model.coef_[0]
    top = np.argsort(np.abs(weights))[::-1][:8]
    return ProbeResult(
        concept=concept,
        model=model,
        scaler=scaler,
        auroc=_safe_auroc(eval_labels, score),
        average_precision=_safe_ap(eval_labels, score),
        sparsity=int(np.sum(np.abs(weights) > 1e-8)),
        top_dims=[int(i) for i in top],
        top_weights=[float(weights[i]) for i in top],
    )


def train_probes(
    train_hidden: np.ndarray,
    train_labels: dict[str, np.ndarray],
    eval_hidden: np.ndarray,
    eval_labels: dict[str, np.ndarray],
    concepts: list[str],
) -> dict[str, ProbeResult]:
    return {
        concept: train_sparse_probe(
            concept,
            train_hidden,
            train_labels[concept],
            eval_hidden,
            eval_labels[concept],
        )
        for concept in concepts
    }


def unsafe_probability(
    probes: dict[str, ProbeResult],
    hidden: np.ndarray,
    concepts: list[str] | None = None,
) -> np.ndarray:
    names = concepts or ["collision_risk", "wrong_object", "low_clearance", "semantic_hazard"]
    probs = np.stack([probes[name].predict_proba(hidden) for name in names], axis=1)
    return 1.0 - np.prod(1.0 - probs, axis=1)
