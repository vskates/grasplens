from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grasplens.visualization import plot_activation_effect, plot_metric_bars, plot_probe_aurocs


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    result = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_bars(result["metrics"], "unsafe_grasp_rate", "Unsafe grasp rate", figures_dir / "unsafe_rate_by_method.png")
    plot_metric_bars(result["metrics"], "success_rate", "Success rate", figures_dir / "success_rate_by_method.png")
    plot_probe_aurocs(result["probes"], figures_dir / "probe_auroc_by_concept.png")
    plot_activation_effect(result["activation_patching"], figures_dir / "activation_patching_effect.png")


if __name__ == "__main__":
    main()
