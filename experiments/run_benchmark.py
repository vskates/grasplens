from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grasplens.conformal import fit_binary_conformal
from grasplens.dataset import build_dataset
from grasplens.filter import select_all
from grasplens.metrics import activation_patching_effect, evaluate_selections
from grasplens.policy import TinyGraspScorer
from grasplens.probes import train_probes, unsafe_probability
from grasplens.scene import generate_scenes
from grasplens.visualization import (
    plot_activation_effect,
    plot_example_scene,
    plot_metric_bars,
    plot_probe_aurocs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GraspLens synthetic benchmark.")
    parser.add_argument("--train-scenes", type=int, default=2000)
    parser.add_argument("--test-scenes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates-per-object", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--patch-alpha", type=float, default=1.5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    return parser.parse_args()


def _json_float(value: float) -> float | None:
    value = float(value)
    if np.isnan(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic train/test scenes...")
    train_scenes = generate_scenes(args.train_scenes, "train", args.seed)
    test_scenes = generate_scenes(args.test_scenes, "test", args.seed + 10_000)

    print("Building candidate datasets...")
    train_data = build_dataset(
        train_scenes,
        seed=args.seed + 1,
        candidates_per_object=args.candidates_per_object,
        progress_desc="train dataset",
    )
    test_data = build_dataset(
        test_scenes,
        seed=args.seed + 2,
        candidates_per_object=args.candidates_per_object,
        progress_desc="test dataset",
    )

    print("Training learned grasp scorer...")
    scorer = TinyGraspScorer(train_data.x.shape[1], hidden_dim=args.hidden_dim, seed=args.seed)
    history = scorer.fit(train_data.x, train_data.y, epochs=args.epochs)
    train_scores, train_hidden = scorer.predict_with_hidden(train_data.x)
    test_scores, test_hidden = scorer.predict_with_hidden(test_data.x)

    rng = np.random.default_rng(args.seed + 3)
    order = rng.permutation(len(train_data.x))
    split = int(0.72 * len(order))
    probe_train_idx = order[:split]
    calib_idx = order[split:]

    concepts = ["collision_risk", "wrong_object", "low_clearance", "semantic_hazard", "unsafe"]
    print("Training sparse concept probes...")
    probe_train_labels = {name: values[probe_train_idx] for name, values in train_data.labels.items()}
    test_labels = test_data.labels
    probes = train_probes(
        train_hidden[probe_train_idx],
        probe_train_labels,
        test_hidden,
        test_labels,
        concepts,
    )

    calib_prob = unsafe_probability(probes, train_hidden[calib_idx])
    test_unsafe_prob = unsafe_probability(probes, test_hidden)
    conformal = fit_binary_conformal(
        calib_prob,
        train_data.labels["unsafe"][calib_idx],
        alpha=args.alpha,
    )
    conformal_coverage = conformal.coverage(test_unsafe_prob, test_data.labels["unsafe"])
    safe_calib = calib_prob[train_data.labels["unsafe"][calib_idx] == 0]
    risk_threshold = float(np.quantile(safe_calib, 0.82)) if len(safe_calib) else 0.42
    risk_threshold = float(np.clip(risk_threshold, 0.18, 0.62))

    print("Evaluating runtime filters...")
    selections = select_all(
        test_data,
        test_scores,
        test_unsafe_prob,
        conformal,
        risk_threshold=risk_threshold,
    )
    metric_rows = evaluate_selections(test_data, selections, test_scores)
    metrics = [row.__dict__ for row in metric_rows]

    unsafe_direction = probes["unsafe"].direction
    patched_scores = scorer.predict_from_hidden(test_hidden + args.patch_alpha * unsafe_direction)
    patch_effect = activation_patching_effect(test_data, test_scores, test_hidden, patched_scores)

    probes_json = {
        name: {
            "auroc": _json_float(result.auroc),
            "average_precision": _json_float(result.average_precision),
            "sparsity": result.sparsity,
            "top_dims": result.top_dims,
            "top_weights": result.top_weights,
        }
        for name, result in probes.items()
    }
    result = {
        "project": "GraspLens",
        "seed": args.seed,
        "train_scenes": args.train_scenes,
        "test_scenes": args.test_scenes,
        "candidates_per_object": args.candidates_per_object,
        "num_train_candidates": int(len(train_data.x)),
        "num_test_candidates": int(len(test_data.x)),
        "policy_backend": history.backend,
        "policy_final_loss": float(history.losses[-1]),
        "policy_losses": [float(x) for x in history.losses],
        "conformal": {
            "alpha": args.alpha,
            "qhat": conformal.qhat,
            "test_coverage": conformal_coverage,
            "risk_threshold": risk_threshold,
        },
        "metrics": metrics,
        "probes": probes_json,
        "activation_patching": patch_effect,
    }

    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(metrics).to_csv(args.output_dir / "metrics.csv", index=False)

    print("Rendering figures...")
    plot_metric_bars(metrics, "unsafe_grasp_rate", "Unsafe grasp rate", figures_dir / "unsafe_rate_by_method.png")
    plot_metric_bars(metrics, "success_rate", "Success rate", figures_dir / "success_rate_by_method.png")
    plot_probe_aurocs(probes_json, figures_dir / "probe_auroc_by_concept.png")
    plot_activation_effect(patch_effect, figures_dir / "activation_patching_effect.png")
    plot_example_scene(
        test_data,
        selections,
        test_scores,
        test_unsafe_prob,
        figures_dir / "example_scene.png",
    )

    print(f"Wrote {results_path}")
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
