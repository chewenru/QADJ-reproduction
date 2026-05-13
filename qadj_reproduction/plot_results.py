import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qadj")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SACRED_ROOT = ROOT / "logs" / "sacred"
FIG_ROOT = ROOT / "figures"
TABLE_ROOT = ROOT / "tables"


def load_runs():
    records = []
    for run_dir in sorted(SACRED_ROOT.glob("*")):
        config_path = run_dir / "config.json"
        info_path = run_dir / "info.json"
        if not config_path.exists() or not info_path.exists():
            continue
        config = json.loads(config_path.read_text())
        info = json.loads(info_path.read_text())
        records.append({"run_dir": run_dir, "config": config, "info": info})
    return records


def parse_run(record):
    method = record["config"].get("algo_name", record["config"].get("name", ""))
    env = record["config"].get("env_label")
    if not env:
        env = record["config"].get("env_args", {}).get("scenario_name") or record["config"].get("env_args", {}).get("map_name")
    if not env:
        env = record["config"].get("env")
    info = record["info"]
    metric_key = "test_battle_won_mean" if "test_battle_won_mean" in info else "test_return_mean"
    ts_key = f"{metric_key}_T"
    if metric_key not in info or ts_key not in info:
        return None
    def normalize_metric(value):
        if isinstance(value, dict) and "value" in value:
            return float(value["value"])
        return float(value)

    return pd.DataFrame(
        {
            "t_env": info[ts_key],
            "metric": [normalize_metric(v) for v in info[metric_key]],
            "method": method,
            "env": env,
            "seed": record["config"].get("seed"),
        }
    )


def save_raw_and_summary(all_curves):
    raw_df = pd.concat(all_curves, ignore_index=True)
    raw_df.to_csv(TABLE_ROOT / "raw_curves.csv", index=False)

    final_rows = []
    for (env, method, seed), df in raw_df.groupby(["env", "method", "seed"]):
        final_rows.append(
            {
                "env": env,
                "method": method,
                "seed": seed,
                "final_metric": df.sort_values("t_env").iloc[-1]["metric"],
            }
        )
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(TABLE_ROOT / "final_per_seed.csv", index=False)
    summary = (
        final_df.groupby(["env", "method"])["final_metric"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "final_mean", "std": "final_std", "count": "n_seeds"})
    )
    summary.to_csv(TABLE_ROOT / "final_summary.csv", index=False)
    return raw_df


def plot_group(raw_df, env, ylabel):
    plt.figure(figsize=(8, 5))
    for method, df in sorted(raw_df[raw_df["env"] == env].groupby("method")):
        grouped = df.groupby("t_env")["metric"]
        mean = grouped.mean()
        std = grouped.std().fillna(0)
        x = mean.index.to_list()
        y = mean.to_list()
        y1 = (mean - std).to_list()
        y2 = (mean + std).to_list()
        plt.plot(x, y, label=method)
        plt.fill_between(x, y1, y2, alpha=0.2)
    plt.xlabel("environment timesteps")
    plt.ylabel(ylabel)
    plt.title(env)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_ROOT / f"{env}.png", dpi=200)
    plt.close()


def main():
    FIG_ROOT.mkdir(exist_ok=True)
    TABLE_ROOT.mkdir(exist_ok=True)
    records = load_runs()
    curves = [df for df in (parse_run(record) for record in records) if df is not None]
    if not curves:
        raise SystemExit("No runnable Sacred logs found.")
    raw_df = save_raw_and_summary(curves)
    for env in sorted(raw_df["env"].unique()):
        ylabel = "win rate" if "vs" in env else "reward"
        plot_group(raw_df, env, ylabel)


if __name__ == "__main__":
    main()
