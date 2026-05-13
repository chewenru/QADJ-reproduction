import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qadj")

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parent.parent
TB_ROOT = ROOT / "results" / "tb_logs"
FIG_ROOT = ROOT / "figures"
TABLE_ROOT = ROOT / "tables"


# Fixed paper-facing selections.
RUN_SPECS = [
    {
        "panel": "QPLEX on 5m_vs_6m",
        "env": "5m_vs_6m",
        "metric": "test_battle_won_mean",
        "ylabel": "Win Rate",
        "scale": 1.0,
        "runs": [
            (
                "QPLEX",
                "supp_qplex_5m_vs_6m_seed1_official_smac__2026-04-28_16-48-28",
            ),
            (
                "QADJ-QPLEX",
                "supp_qadj_qplex_5m_vs_6m_seed1_official_smac__2026-04-28_16-48-45",
            ),
        ],
    },
    {
        "panel": "QTRAN on Predator-prey",
        "env": "Predator-prey",
        "metric": "test_return_mean",
        "ylabel": "Scaled Test Return Mean",
        "scale": 6.0,
        "runs": [
            (
                "QTRAN",
                "supp_qtran_predator_seed2_paper_2_wsl__2026-04-23_17-58-20",
            ),
            (
                "QADJ-QTRAN",
                "supp_qadj_qtran_paper_predator_seed2_screen_bs2_700k__2026-04-24_10-22-47",
            ),
        ],
    },
]


def load_scalar_series(run_dir: Path, tag: str, scale: float):
    event_acc = EventAccumulator(str(run_dir))
    event_acc.Reload()
    available = event_acc.Tags().get("scalars", [])
    if tag not in available:
        raise ValueError(f"{run_dir.name} does not contain scalar tag {tag!r}. Available: {available}")
    scalars = event_acc.Scalars(tag)
    x = [item.step for item in scalars]
    y = [item.value * scale for item in scalars]
    return x, y


def ema_smooth(values, smoothing=0.90):
    if not values:
        return values
    smoothed = [values[0]]
    weight = float(smoothing)
    for value in values[1:]:
        smoothed.append(smoothed[-1] * weight + value * (1.0 - weight))
    return smoothed


def plot_selected_extensions():
    FIG_ROOT.mkdir(exist_ok=True)
    TABLE_ROOT.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    summary_rows = []

    for ax, spec in zip(axes, RUN_SPECS):
        for label, run_name in spec["runs"]:
            run_dir = TB_ROOT / run_name
            if not run_dir.exists():
                raise FileNotFoundError(f"Missing TensorBoard log directory: {run_dir}")
            x, y = load_scalar_series(run_dir, spec["metric"], spec["scale"])
            smooth_y = ema_smooth(y, smoothing=0.90)
            raw_line = ax.plot(x, y, linewidth=1.0, alpha=0.22)[0]
            ax.plot(x, smooth_y, linewidth=2.6, label=label, color=raw_line.get_color())
            summary_rows.append(
                {
                    "panel": spec["panel"],
                    "environment": spec["env"],
                    "method": label,
                    "run_name": run_name,
                    "metric": spec["metric"],
                    "scale_factor": spec["scale"],
                    "final_value": y[-1],
                    "best_value": max(y),
                    "last_step": x[-1],
                }
            )

        ax.set_title(spec["panel"])
        ax.set_xlabel("Environment Timesteps")
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, alpha=0.25)
        ax.legend()

    plt.tight_layout()
    fig_path = FIG_ROOT / "paper_extension_selected.png"
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    csv_path = TABLE_ROOT / "paper_extension_selected_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "panel",
                "environment",
                "method",
                "run_name",
                "metric",
                "scale_factor",
                "final_value",
                "best_value",
                "last_step",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved figure to {fig_path}")
    print(f"Saved summary to {csv_path}")


if __name__ == "__main__":
    plot_selected_extensions()
