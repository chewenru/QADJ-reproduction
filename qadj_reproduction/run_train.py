import argparse
import itertools
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

MAIN_METHODS = {
    "vdn": "vdn",
    "qmix": "qmix",
    "qtran": "qtran",
    "wqmix": "wqmix",
    "qplex": "qplex",
    "qadj_vdn": "qadj_vdn",
    "qadj_qmix": "qadj_qmix",
}

SUPPLEMENT_METHODS = {
    "qplex": "qplex",
    "qadj_qplex": "qadj_qplex",
}

SUPPLEMENT_QTRAN_METHODS = {
    "qtran": "qtran",
    "qadj_qtran": "qadj_qtran",
}

MPE_ENVS = {
    "predator": {"env_config": "mpe", "env_args.scenario_name": "predator_prey", "t_max": 2050000},
    "nav": {"env_config": "mpe", "env_args.scenario_name": "cooperative_navigation", "t_max": 2050000},
    "keep_away": {"env_config": "mpe", "env_args.scenario_name": "keep_away", "t_max": 2050000},
}

SMAC_ENVS = {
    "5m_vs_6m": {"env_config": "sc2", "env_args.map_name": "5m_vs_6m", "env_args.difficulty": 7, "t_max": 2000050},
    "2c_vs_64zg": {"env_config": "sc2", "env_args.map_name": "2c_vs_64zg", "env_args.difficulty": 7, "t_max": 2000050},
}

DEFAULT_OVERRIDES = {
    "lr": 0.0005,
    "gamma": 0.99,
    "buffer_size": 5000,
    "batch_size": 32,
    "epsilon_start": 1.0,
    "epsilon_finish": 0.05,
    "epsilon_anneal_time": 50000,
    "target_update_interval": 200,
    "test_interval": 10000,
    "test_nepisode": 100,
    "save_model": True,
    "save_model_interval": 50000,
    "use_tensorboard": True,
    "log_interval": 1000,
    "runner_log_interval": 1000,
    "learner_log_interval": 1000,
}


def fmt_override(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def build_command(config_name, env_name, seed, run_name, extra_overrides):
    env_def = MPE_ENVS.get(env_name, SMAC_ENVS.get(env_name))
    overrides = dict(DEFAULT_OVERRIDES)
    overrides.update(env_def)
    overrides.update(extra_overrides)
    overrides["seed"] = seed
    overrides["name"] = run_name
    overrides["algo_name"] = config_name
    overrides["env_label"] = env_name

    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        f"--config={config_name}",
        f"--env-config={env_def['env_config']}",
        "with",
    ]
    for key, value in overrides.items():
        if key == "env_config":
            continue
        cmd.append(f"{key}={fmt_override(value)}")
    return cmd


def resolve_method_config(method, env_name):
    if method == "qplex":
        return "qplex_smac" if env_name in SMAC_ENVS else "qplex"
    if method == "qadj_qplex":
        return "qadj_qplex_smac" if env_name in SMAC_ENVS else "qadj_qplex"
    return MAIN_METHODS.get(method, SUPPLEMENT_METHODS.get(method, method))


def iter_main_runs(seeds):
    for method in MAIN_METHODS:
        for env_name in itertools.chain(MPE_ENVS.keys(), SMAC_ENVS.keys()):
            for seed in seeds:
                run_name = f"main_{method}_{env_name}_seed{seed}"
                yield build_command(resolve_method_config(method, env_name), env_name, seed, run_name, {})


def iter_fig9_runs(seeds):
    mpe_methods = dict(MAIN_METHODS)
    for method, config_name in mpe_methods.items():
        for env_name in MPE_ENVS.keys():
            for seed in seeds:
                run_name = f"main_{method}_{env_name}_seed{seed}"
                yield build_command(config_name, env_name, seed, run_name, {})


def iter_fig10_runs(seeds):
    for method in MAIN_METHODS:
        for env_name in SMAC_ENVS.keys():
            for seed in seeds:
                run_name = f"main_{method}_{env_name}_seed{seed}"
                yield build_command(resolve_method_config(method, env_name), env_name, seed, run_name, {})


def iter_t2_ablation(seeds):
    for t2 in [200, 400, 800, 1200, 1600, 2000]:
        for seed in seeds:
            run_name = f"ablation_t2_{t2}_predator_seed{seed}"
            yield build_command("qadj_qmix", "predator", seed, run_name, {"qadj_t2": t2})


def iter_rho_ablation(seeds):
    for rho in [0.1, 0.3, 0.5, 0.7]:
        for seed in seeds:
            run_name = f"ablation_rho_{rho}_predator_seed{seed}"
            yield build_command("qadj_qmix", "predator", seed, run_name, {"qadj_rho": rho})


def iter_bounds_ablation(seeds):
    configs = [
        ("ablation_no_upper", {"qadj_enable_upper_bound": False}),
        ("ablation_no_lower", {"qadj_enable_lower_bound": False}),
    ]
    for label, overrides in configs:
        for seed in seeds:
            run_name = f"{label}_predator_seed{seed}"
            yield build_command("qadj_qmix", "predator", seed, run_name, overrides)


def iter_control_ablation(seeds):
    for seed in seeds:
        run_name = f"ablation_control_on_predator_seed{seed}"
        yield build_command("qadj_qmix", "predator", seed, run_name, {"qadj_control_bounds": True})
        run_name = f"ablation_control_off_predator_seed{seed}"
        yield build_command("qadj_qmix", "predator", seed, run_name, {"qadj_control_bounds": False})


def iter_qadj_qplex_bridge(seeds):
    for method in SUPPLEMENT_METHODS:
        for env_name in ("predator", "5m_vs_6m"):
            for seed in seeds:
                run_name = f"supp_{method}_{env_name}_seed{seed}"
                yield build_command(resolve_method_config(method, env_name), env_name, seed, run_name, {})


def iter_qadj_qtran_bridge(seeds):
    for method, config_name in SUPPLEMENT_QTRAN_METHODS.items():
        for env_name in ("predator", "5m_vs_6m"):
            for seed in seeds:
                run_name = f"supp_{method}_{env_name}_seed{seed}"
                yield build_command(config_name, env_name, seed, run_name, {})


SUITES = {
    "main": iter_main_runs,
    "fig9_mpe_main": iter_fig9_runs,
    "fig10_smac_main": iter_fig10_runs,
    "ablation_t2": iter_t2_ablation,
    "ablation_rho": iter_rho_ablation,
    "ablation_bounds": iter_bounds_ablation,
    "ablation_control": iter_control_ablation,
    "qadj_qplex_bridge": iter_qadj_qplex_bridge,
    "qadj_qtran_bridge": iter_qadj_qtran_bridge,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=list(SUITES) + ["all"], default="main")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ROOT.parent.joinpath("logs").mkdir(exist_ok=True)
    ROOT.parent.joinpath("checkpoints").mkdir(exist_ok=True)
    ROOT.parent.joinpath("figures").mkdir(exist_ok=True)
    ROOT.parent.joinpath("tables").mkdir(exist_ok=True)

    suites = list(SUITES) if args.suite == "all" else [args.suite]
    commands = []
    for suite in suites:
        commands.extend(SUITES[suite](args.seeds))

    for cmd in commands:
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=ROOT.parent)


if __name__ == "__main__":
    main()
