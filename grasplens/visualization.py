from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from grasplens.dataset import BenchmarkDataset
from grasplens.filter import Selection
from grasplens.geometry import grasp_polygon


COLOR_MAP = {
    "red": "#d84a3a",
    "blue": "#3b6fb6",
    "green": "#4f9d69",
    "yellow": "#d9b84f",
    "purple": "#8d67b1",
}


def _scene_selection_map(selections: list[Selection], dataset: BenchmarkDataset) -> dict[int, dict[str, Selection]]:
    result: dict[int, dict[str, Selection]] = {}
    for selection in selections:
        scene_idx = int(dataset.scene_index[selection.index])
        result.setdefault(scene_idx, {})[selection.method] = selection
    return result


def choose_interesting_scene(dataset: BenchmarkDataset, selections: list[Selection]) -> int:
    by_scene = _scene_selection_map(selections, dataset)
    for scene_idx, methods in by_scene.items():
        baseline = methods.get("baseline")
        full = methods.get("grasplens_full")
        if baseline is None or full is None:
            continue
        if dataset.predicates[baseline.index].unsafe and dataset.predicates[full.index].success:
            return scene_idx
    for scene_idx, methods in by_scene.items():
        baseline = methods.get("baseline")
        if baseline is not None and dataset.predicates[baseline.index].unsafe:
            return scene_idx
    return 0


def plot_example_scene(
    dataset: BenchmarkDataset,
    selections: list[Selection],
    scores: np.ndarray,
    unsafe_prob: np.ndarray,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene_idx = choose_interesting_scene(dataset, selections)
    scene = dataset.scenes[scene_idx]
    indices = np.flatnonzero(dataset.scene_index == scene_idx)
    selected = _scene_selection_map(selections, dataset)[scene_idx]

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    x0, y0, x1, y1 = scene.shelf_bounds
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2.0, ec="#2b2b2b"))

    for obs in scene.obstacles:
        ox0, oy0, ox1, oy1 = obs.bounds
        ax.add_patch(
            Rectangle((ox0, oy0), ox1 - ox0, oy1 - oy0, fc="#6b6b6b", ec="#333333", alpha=0.75)
        )

    for obj in scene.objects:
        ox0, oy0, ox1, oy1 = obj.bounds
        edge = "#111111" if obj.is_target else "#6f6f6f"
        lw = 2.6 if obj.is_target else 1.2
        hatch = "///" if obj.fragile or obj.hazard else None
        ax.add_patch(
            Rectangle(
                (ox0, oy0),
                ox1 - ox0,
                oy1 - oy0,
                fc=COLOR_MAP[obj.color],
                ec=edge,
                lw=lw,
                alpha=0.86,
                hatch=hatch,
            )
        )
        ax.text(obj.x, obj.y, "target" if obj.is_target else "distractor", ha="center", va="center", fontsize=8)

    top_by_score = indices[np.argsort(scores[indices])[::-1][:16]]
    for idx in top_by_score:
        pred = dataset.predicates[idx]
        color = "#c44e52" if pred.unsafe else "#4c72b0"
        ax.add_patch(
            Polygon(
                grasp_polygon(dataset.candidates[idx]),
                closed=True,
                fill=False,
                lw=1.0,
                ec=color,
                alpha=0.35 + 0.35 * float(unsafe_prob[idx] > 0.5),
            )
        )

    styles = {
        "baseline": ("#b00020", "baseline"),
        "geometry": ("#cc7a00", "geometry"),
        "probe": ("#7d4cc2", "probe"),
        "grasplens_full": ("#087f5b", "GraspLens full"),
    }
    for method, (color, label) in styles.items():
        if method not in selected:
            continue
        idx = selected[method].index
        ax.add_patch(
            Polygon(
                grasp_polygon(dataset.candidates[idx]),
                closed=True,
                fill=False,
                lw=3.0 if method in {"baseline", "grasplens_full"} else 2.0,
                ec=color,
                label=f"{label}: risk={unsafe_prob[idx]:.2f}",
            )
        )

    ax.set_title("Example shifted shelf scene: unsafe learned choice vs filtered grasp")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("shelf x")
    ax.set_ylabel("shelf y")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_bars(metrics: list[dict], key: str, ylabel: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    methods = [row["method"] for row in metrics]
    values = [row[key] for row in metrics]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    colors = ["#b00020", "#cc7a00", "#7d4cc2", "#087f5b"]
    ax.bar(methods, values, color=colors[: len(methods)])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, max(1.0, max(values) * 1.18))
    ax.tick_params(axis="x", rotation=15)
    for i, value in enumerate(values):
        ax.text(i, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_probe_aurocs(probes: dict[str, dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [name for name in probes if name != "unsafe"]
    values = [probes[name]["auroc"] for name in names]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(names, values, color="#4c72b0")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(axis="x", rotation=20)
    for i, value in enumerate(values):
        ax.text(i, min(value + 0.02, 0.98), f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_activation_effect(effect: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    keys = ["unsafe_score_delta", "unsafe_rank_delta_positive_means_moves_up"]
    values = [effect[k] for k in keys]
    ax.bar(["score delta", "rank delta"], values, color=["#4c72b0", "#c44e52"])
    ax.axhline(0.0, color="#333333", lw=1.0)
    ax.set_title("Activation steering along unsafe concept")
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

