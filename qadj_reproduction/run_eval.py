import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--test-nepisode", type=int, default=100)
    args = parser.parse_args()

    sacred_root = ROOT.parent / "logs" / "sacred"
    matches = []
    for run_dir in sacred_root.iterdir():
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        if config.get("name") == args.run_name:
            matches.append((run_dir, config))

    if not matches:
        raise SystemExit(f"Run {args.run_name} not found under {sacred_root}")

    run_dir, config = sorted(matches, key=lambda item: item[0].name)[-1]
    checkpoint_root = ROOT.parent / "checkpoints" / "models"
    ckpt_dir = checkpoint_root / config["unique_token"]
    if not ckpt_dir.exists():
        raise SystemExit(f"Checkpoint dir missing: {ckpt_dir}")

    algo_name = config.get("algo_name")
    if not algo_name:
        raise SystemExit(f"algo_name missing in config: {run_dir / 'config.json'}")

    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        f"--config={algo_name}",
        f"--env-config={config['env']}",
        "with",
        f"checkpoint_path={ckpt_dir}",
        "evaluate=True",
        f"test_nepisode={args.test_nepisode}",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT.parent)


if __name__ == "__main__":
    main()
