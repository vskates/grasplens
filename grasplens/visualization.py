from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from grasplens.dataset import BenchmarkDataset
from grasplens.filter import Selection
from grasplens.geometry import grasp_polygon


INK = "#17201d"
MUTED = "#65716c"
LINE = "#d6ddd8"
PAPER = "#f7faf7"
PANEL = "#ffffff"
GREEN = "#087f5b"
GREEN_LIGHT = "#dff2ea"
RED = "#b21d3a"
RED_LIGHT = "#f8e2e7"
AMBER = "#c8751a"
VIOLET = "#7455a6"
BLUE = "#386f99"
OBSTACLE = "#707873"

COLOR_MAP = {
    "red": "#d95849",
    "blue": "#4b78b7",
    "green": "#5aa574",
    "yellow": "#d4b44b",
    "purple": "#8b6ab2",
}

METHOD_LABELS = {
    "baseline": "Baseline",
    "geometry": "Geometry",
    "probe": "Probe",
    "grasplens_full": "GraspLens full",
}

METHOD_COLORS = {
    "baseline": RED,
    "geometry": AMBER,
    "probe": VIOLET,
    "grasplens_full": GREEN,
}


def _apply_theme() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.titleweight": 800,
            "axes.labelsize": 10,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": LINE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": PAPER,
            "axes.facecolor": PANEL,
            "savefig.facecolor": PAPER,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
        }
    )


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


def _rounded_rect(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    edge: str,
    lw: float = 1.4,
    alpha: float = 1.0,
    hatch: str | None = None,
    radius: float = 0.008,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        fc=color,
        ec=edge,
        lw=lw,
        alpha=alpha,
        hatch=hatch,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    return patch


def _draw_label(ax: plt.Axes, x: float, y: float, text: str, fc: str, color: str = INK) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight=800,
        color=color,
        bbox={
            "boxstyle": "round,pad=0.24,rounding_size=0.08",
            "fc": fc,
            "ec": "none",
            "alpha": 0.94,
        },
        zorder=9,
    )


def _predicate_summary(dataset: BenchmarkDataset, idx: int) -> str:
    pred = dataset.predicates[idx]
    issues = []
    if pred.wrong_object:
        issues.append("wrong object")
    if pred.low_clearance:
        issues.append("low clearance")
    if pred.collision:
        issues.append("collision")
    if pred.semantic_hazard:
        issues.append("hazard")
    if pred.success:
        return "target contact, predicates pass"
    return ", ".join(issues) if issues else "not successful"


def _draw_card(
    ax: plt.Axes,
    xy: tuple[float, float],
    title: str,
    value: str,
    body: str,
    color: str,
    fill: str,
) -> None:
    x, y = xy
    card = FancyBboxPatch(
        (x, y),
        0.92,
        0.22,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        fc=fill,
        ec=color,
        lw=1.35,
        transform=ax.transAxes,
    )
    ax.add_patch(card)
    ax.text(x + 0.04, y + 0.165, title, transform=ax.transAxes, color=color, fontsize=10, fontweight=850)
    ax.text(x + 0.04, y + 0.103, value, transform=ax.transAxes, color=INK, fontsize=16, fontweight=850)
    ax.text(x + 0.04, y + 0.045, body, transform=ax.transAxes, color=MUTED, fontsize=8.6, wrap=True)


def plot_example_scene(
    dataset: BenchmarkDataset,
    selections: list[Selection],
    scores: np.ndarray,
    unsafe_prob: np.ndarray,
    output_path: str | Path,
) -> None:
    _apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene_idx = choose_interesting_scene(dataset, selections)
    scene = dataset.scenes[scene_idx]
    indices = np.flatnonzero(dataset.scene_index == scene_idx)
    selected = _scene_selection_map(selections, dataset)[scene_idx]

    fig = plt.figure(figsize=(11.4, 7.0), dpi=190)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 0.82], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    audit = fig.add_subplot(gs[0, 1])

    fig.suptitle(
        "Shifted shelf scene: learned shortcut vs runtime assurance",
        x=0.055,
        y=0.982,
        ha="left",
        color=INK,
        fontsize=18,
        fontweight=900,
    )
    fig.text(
        0.055,
        0.94,
        "Red objects are no longer reliable target cues. The baseline follows the shortcut; GraspLens filters it.",
        ha="left",
        color=MUTED,
        fontsize=10.4,
    )

    x0, y0, x1, y1 = scene.shelf_bounds
    ax.set_facecolor("#f9fbfa")
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=True,
            fc="#fbfdfb",
            ec=INK,
            lw=2.4,
            zorder=1,
        )
    )
    for tick in np.arange(0.2, 0.9, 0.2):
        ax.plot([x0, x1], [tick, tick], color="#eef2ef", lw=0.8, zorder=0)
        ax.plot([tick, tick], [y0, y1], color="#eef2ef", lw=0.8, zorder=0)

    for obs_id, obs in enumerate(scene.obstacles, start=1):
        ox0, oy0, ox1, oy1 = obs.bounds
        _rounded_rect(ax, ox0, oy0, ox1 - ox0, oy1 - oy0, OBSTACLE, "#3c4340", lw=1.2, alpha=0.86)
        _draw_label(ax, obs.x, obs.y, f"O{obs_id}", "#edf0ee", color="#353b38")

    distractor_id = 1
    for obj in scene.objects:
        ox0, oy0, ox1, oy1 = obj.bounds
        edge = INK if obj.is_target else "#626b66"
        lw = 2.4 if obj.is_target else 1.15
        hatch = "///" if obj.fragile or obj.hazard else None
        _rounded_rect(
            ax,
            ox0,
            oy0,
            ox1 - ox0,
            oy1 - oy0,
            COLOR_MAP[obj.color],
            edge,
            lw=lw,
            alpha=0.88,
            hatch=hatch,
        )
        if obj.is_target:
            label = "TARGET"
            fill = "#e7f4ec"
        else:
            label = f"D{distractor_id}"
            fill = "#fff5ec" if obj.color == "red" else "#eef2ef"
            distractor_id += 1
        _draw_label(ax, obj.x, obj.y, label, fill)

    top_by_score = indices[np.argsort(scores[indices])[::-1][:18]]
    for idx in top_by_score:
        pred = dataset.predicates[idx]
        color = RED if pred.unsafe else BLUE
        alpha = 0.18 + 0.34 * float(unsafe_prob[idx] > 0.55)
        ax.add_patch(
            Polygon(
                grasp_polygon(dataset.candidates[idx]),
                closed=True,
                fill=False,
                lw=1.15,
                ec=color,
                alpha=alpha,
                zorder=6,
            )
        )

    selected_styles = {
        "baseline": (RED, 4.0, "Baseline selected"),
        "grasplens_full": (GREEN, 4.0, "GraspLens selected"),
        "geometry": (AMBER, 2.4, "Geometry"),
        "probe": (VIOLET, 2.4, "Probe"),
    }
    for method in ("geometry", "probe", "baseline", "grasplens_full"):
        if method not in selected:
            continue
        color, lw, label = selected_styles[method]
        idx = selected[method].index
        poly = Polygon(
            grasp_polygon(dataset.candidates[idx]),
            closed=True,
            fill=False,
            lw=lw,
            ec=color,
            alpha=0.98,
            zorder=10 if method in {"baseline", "grasplens_full"} else 8,
            label=label,
        )
        poly.set_path_effects([pe.Stroke(linewidth=lw + 2.4, foreground="white", alpha=0.8), pe.Normal()])
        ax.add_patch(poly)

    baseline_idx = selected["baseline"].index
    full_idx = selected["grasplens_full"].index
    for idx, text, color, dx, dy in [
        (baseline_idx, "rejected", RED, 0.05, 0.06),
        (full_idx, "accepted", GREEN, 0.045, 0.065),
    ]:
        cand = dataset.candidates[idx]
        ax.annotate(
            text,
            xy=(cand.x, cand.y),
            xytext=(min(cand.x + dx, 0.86), min(cand.y + dy, 0.9)),
            color=color,
            fontsize=9,
            fontweight=850,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5},
            bbox={"boxstyle": "round,pad=0.25,rounding_size=0.08", "fc": "white", "ec": color, "lw": 1.0},
            zorder=14,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("shelf x")
    ax.set_ylabel("shelf y")
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor=LINE,
        fontsize=8.6,
    )

    audit.set_axis_off()
    audit.set_facecolor(PAPER)
    audit.text(0.04, 0.92, "Decision audit", transform=audit.transAxes, color=INK, fontsize=17, fontweight=900)
    audit.text(
        0.04,
        0.858,
        "The learned score is treated as utility, not as a safety certificate.",
        transform=audit.transAxes,
        color=MUTED,
        fontsize=9.6,
        wrap=True,
    )
    _draw_card(
        audit,
        (0.04, 0.59),
        "Baseline",
        f"risk {unsafe_prob[baseline_idx]:.2f}",
        _predicate_summary(dataset, baseline_idx),
        RED,
        RED_LIGHT,
    )
    _draw_card(
        audit,
        (0.04, 0.325),
        "GraspLens full",
        f"risk {unsafe_prob[full_idx]:.2f}",
        _predicate_summary(dataset, full_idx),
        GREEN,
        GREEN_LIGHT,
    )
    audit.text(0.04, 0.205, "Runtime rule", transform=audit.transAxes, color=INK, fontsize=10, fontweight=850)
    audit.text(
        0.04,
        0.12,
        "Select the highest learned score after hard geometry predicates and calibrated latent-risk rejection.",
        transform=audit.transAxes,
        color=MUTED,
        fontsize=9.1,
        wrap=True,
    )
    audit.text(
        0.04,
        0.025,
        f"scene {scene_idx} · {len(indices)} candidates",
        transform=audit.transAxes,
        color="#88928d",
        fontsize=8.8,
        fontweight=700,
    )

    fig.savefig(output_path)
    plt.close(fig)


def _clean_axis(ax: plt.Axes, xlim: tuple[float, float] = (0.0, 1.0)) -> None:
    ax.set_xlim(*xlim)
    ax.grid(axis="x", color="#e4eae6", lw=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)


def plot_metric_bars(metrics: list[dict], key: str, ylabel: str, output_path: str | Path) -> None:
    _apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    methods = [row["method"] for row in metrics]
    labels = [METHOD_LABELS[m] for m in methods]
    values = np.asarray([row[key] for row in metrics], dtype=float)
    colors = [METHOD_COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=190)
    y = np.arange(len(methods))
    bars = ax.barh(y, values, color=colors, height=0.54, alpha=0.94)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    _clean_axis(ax)
    title = "Unsafe selections by method" if "unsafe" in key else "Successful grasps by method"
    subtitle = "lower is better" if "unsafe" in key else "higher is better"
    ax.set_title(title, loc="left", pad=18)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=MUTED, fontsize=9.6, fontweight=700)
    ax.set_xlabel(ylabel)

    for bar, value, method in zip(bars, values, methods):
        x = float(bar.get_width())
        label_x = min(x + 0.025, 0.965)
        ha = "left" if x < 0.88 else "right"
        label_x = x + 0.025 if ha == "left" else x - 0.025
        color = INK if method != "grasplens_full" else GREEN
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha=ha,
            color=color,
            fontsize=10.6,
            fontweight=850,
        )
    fig.savefig(output_path)
    plt.close(fig)


def plot_probe_aurocs(probes: dict[str, dict], output_path: str | Path) -> None:
    _apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [name for name in probes if name != "unsafe"]
    labels = [name.replace("_", " ") for name in names]
    values = np.asarray([probes[name]["auroc"] for name in names], dtype=float)
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = values[order]

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=190)
    y = np.arange(len(labels))
    ax.barh(y, values, color=BLUE, height=0.54, alpha=0.92)
    ax.axvline(0.5, color="#a8b0ad", lw=1.0, ls="--")
    ax.set_yticks(y, labels)
    _clean_axis(ax, xlim=(0.45, 1.0))
    ax.set_title("Sparse probe discrimination", loc="left", pad=18)
    ax.text(0, 1.02, "AUROC on shifted test candidates", transform=ax.transAxes, color=MUTED, fontsize=9.6, fontweight=700)
    ax.set_xlabel("AUROC")
    for yi, value in zip(y, values):
        ax.text(value - 0.012, yi, f"{value:.3f}", ha="right", va="center", color="white", fontsize=10, fontweight=850)
    fig.savefig(output_path)
    plt.close(fig)


def plot_activation_effect(effect: dict, output_path: str | Path) -> None:
    _apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Unsafe score delta", "Unsafe rank delta"]
    values = np.asarray(
        [
            effect["unsafe_score_delta"],
            effect["unsafe_rank_delta_positive_means_moves_up"],
        ],
        dtype=float,
    )
    colors = [GREEN if value < 0 else RED for value in values]

    fig, ax = plt.subplots(figsize=(7.0, 3.9), dpi=190)
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.54, alpha=0.92)
    ax.axhline(0.0, color=INK, lw=1.1)
    ax.set_xticks(x, labels)
    ax.set_title("Activation steering diagnostic", loc="left", pad=18)
    ax.text(
        0,
        1.02,
        "negative score delta means unsafe candidates move down",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.6,
        fontweight=700,
    )
    ax.grid(axis="y", color="#e4eae6", lw=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    ymin = min(-1.1, float(values.min()) - 0.18)
    ymax = max(0.25, float(values.max()) + 0.18)
    ax.set_ylim(ymin, ymax)
    for xi, value in zip(x, values):
        va = "top" if value < 0 else "bottom"
        offset = -0.04 if value < 0 else 0.04
        ax.text(xi, value + offset, f"{value:.3f}", ha="center", va=va, color=INK, fontsize=10.5, fontweight=850)
    fig.savefig(output_path)
    plt.close(fig)
