# QADJ Reproduction Code

This directory contains the reproduction code for QADJ experiments on MPE and SMAC-style multi-agent reinforcement learning tasks. The codebase is based on the PyMARL training structure and includes baseline algorithms plus QADJ extensions.

## Algorithms

Baselines:

- `iql`
- `vdn`
- `qmix`
- `qtran`
- `wqmix`
- `qplex`
- `coma`

QADJ variants:

- `qadj_vdn`
- `qadj_qmix`
- `qadj_qplex`
- `qadj_qtran`
- `qadj_qtran_paper`
- `qadj_qtran_paper_smac_adapted_v8`

The main paper-style QADJ-QTRAN implementation is in:

- `learners/qadj_qtran_paper_learner.py`

The SMAC-adapted QADJ-QTRAN configuration used for the 5m_vs_6m extension is:

- `config/algs/qadj_qtran_paper_smac_adapted_v8.yaml`

## Directory Structure

```text
qadj_reproduction/
  config/
    algs/        # algorithm configs
    envs/        # environment configs
  controllers/   # multi-agent controller
  components/    # replay buffer, action selectors, schedules
  envs/          # MPE/SMAC environment wrappers
  learners/      # baseline and QADJ learners
  modules/
    agents/      # recurrent agent network
    mixers/      # VDN, QMIX, QTRAN, QPLEX/DMAQ mixers
  runners/       # episode and parallel runners
  main.py        # Sacred entry point
  run.py         # training loop
  run_train.py   # batch command generator
```

## Environment Setup

Create a virtual environment and install dependencies:

```bash
cd /path/to/repo
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r qadj_reproduction/requirements.txt
```

For GPU training, install the PyTorch build matching your CUDA version before installing the remaining dependencies. See the official PyTorch installation page for the correct command.

The helper script `install_deps.sh` installs a CPU PyTorch build and is mainly intended for quick setup checks:

```bash
bash qadj_reproduction/install_deps.sh
source .venv/bin/activate
```

## MPE Setup

MPE dependencies are installed through `pettingzoo[mpe]` from `requirements.txt`. No extra map files are needed.

Check a short MPE run:

```bash
python qadj_reproduction/main.py --config=vdn --env-config=mpe with \
env_args.scenario_name=predator_prey \
seed=1 \
t_max=50000 \
batch_size=32 \
test_nepisode=20 \
test_interval=10000 \
use_tensorboard=True \
name=smoke_vdn_predator_seed1
```

## SMAC Setup

SMAC experiments require StarCraft II and SMAC maps. The install script downloads SC2 4.6.2 and SMAC maps under `3rdparty/StarCraftII`:

```bash
bash qadj_reproduction/install_sc2.sh
export SC2PATH=/path/to/repo/3rdparty/StarCraftII
```

You can check the SMAC environment with:

```bash
python qadj_reproduction/check_smac_env.py
```

## Running Experiments

All experiments use:

```bash
python qadj_reproduction/main.py --config=<algorithm> --env-config=<environment> with <overrides>
```

Results are written to:

- Sacred logs: `logs/sacred/`
- TensorBoard logs: `results/tb_logs/`
- Model checkpoints: `checkpoints/models/`

Start TensorBoard with:

```bash
tensorboard --logdir results/tb_logs
```

## Reproduce Main MPE Runs

VDN on Predator-Prey:

```bash
python qadj_reproduction/main.py --config=vdn --env-config=mpe with \
env_args.scenario_name=predator_prey \
runner=parallel \
batch_size_run=2 \
seed=1 \
t_max=2050000 \
batch_size=32 \
test_nepisode=100 \
test_interval=10000 \
save_model=True \
save_model_interval=200000 \
use_tensorboard=True \
use_cuda=True \
buffer_cpu_only=False \
name=vdn_predator_seed1
```

QADJ-VDN on Predator-Prey:

```bash
python qadj_reproduction/main.py --config=qadj_vdn --env-config=mpe with \
env_args.scenario_name=predator_prey \
runner=parallel \
batch_size_run=2 \
seed=1 \
t_max=2050000 \
batch_size=32 \
test_nepisode=100 \
test_interval=10000 \
save_model=True \
save_model_interval=200000 \
use_tensorboard=True \
use_cuda=True \
buffer_cpu_only=False \
name=qadj_vdn_predator_seed1
```

QMIX and QADJ-QMIX use the same command structure with `--config=qmix` and `--config=qadj_qmix`.

## Reproduce QADJ-QTRAN Supplement

QTRAN baseline on Predator-Prey:

```bash
python qadj_reproduction/main.py --config=qtran --env-config=mpe with \
env_args.scenario_name=predator_prey \
runner=parallel \
batch_size_run=2 \
seed=2 \
t_max=2050000 \
batch_size=32 \
test_nepisode=100 \
test_interval=10000 \
save_model=True \
save_model_interval=200000 \
use_tensorboard=True \
use_cuda=True \
buffer_cpu_only=False \
name=qtran_predator_seed2
```

QADJ-QTRAN paper-style variant on Predator-Prey:

```bash
python qadj_reproduction/main.py --config=qadj_qtran_paper --env-config=mpe with \
env_args.scenario_name=predator_prey \
runner=parallel \
batch_size_run=2 \
seed=2 \
t_max=2050000 \
batch_size=32 \
test_nepisode=100 \
test_interval=10000 \
save_model=True \
save_model_interval=200000 \
use_tensorboard=True \
use_cuda=True \
buffer_cpu_only=False \
name=qadj_qtran_paper_predator_seed2
```

## Reproduce SMAC 5m_vs_6m QTRAN Runs

QTRAN baseline:

```bash
python qadj_reproduction/main.py --config=qtran --env-config=sc2 with \
env_args.map_name=5m_vs_6m \
env_args.difficulty=7 \
seed=1 \
t_max=2000050 \
batch_size=32 \
test_nepisode=100 \
test_interval=10000 \
save_model=True \
save_model_interval=400000 \
use_tensorboard=True \
use_cuda=True \
buffer_cpu_only=False \
name=qtran_5m_vs_6m_seed1
```

QADJ-QTRAN SMAC-adapted variant:

```bash
python qadj_reproduction/main.py --config=qadj_qtran_paper_smac_adapted_v8 --env-config=sc2 with \
env_args.map_name=5m_vs_6m \
env_args.difficulty=7 \
seed=1 \
t_max=2050000 \
batch_size=32 \
test_nepisode=100 \
test_interval=10000 \
save_model=True \
save_model_interval=400000 \
use_tensorboard=True \
use_cuda=True \
buffer_cpu_only=False \
name=qadj_qtran_paper_smac_adapted_v8_5m_vs_6m_seed1
```

## Batch Command Generator

`run_train.py` can generate or execute standard suites:

```bash
python qadj_reproduction/run_train.py --suite main --seeds 1 2 3 --dry-run
```

Remove `--dry-run` to execute the generated commands sequentially.

Available suites include:

- `main`
- `fig9_mpe_main`
- `fig10_smac_main`
- `ablation_t2`
- `ablation_rho`
- `ablation_bounds`
- `ablation_control`
- `qadj_qplex_bridge`
- `qadj_qtran_bridge`

## Plotting

Plotting helpers are provided in:

- `plot_results.py`
- `plot_paper_extensions.py`

Example:

```bash
python qadj_reproduction/plot_results.py
```

Adjust run names inside the plotting scripts to match your TensorBoard log directories.

## Notes for Reproducibility

- Use the same `seed`, `batch_size_run`, `test_nepisode`, and `test_interval` when comparing a baseline with a QADJ variant.
- `batch_size_run` changes the sampling distribution and can affect MARL results, not just wall-clock speed.
- For paper-quality curves, run multiple seeds and plot mean with variance or standard error.
- Do not compare resumed runs with fresh runs unless the resume procedure is explicitly reported.
- Keep SMAC and MPE results separate because the QADJ schedules are task-specific.

## Citation

If you use this code, cite the corresponding QADJ paper and the original algorithm papers for VDN, QMIX, QTRAN, QPLEX, COMA, and SMAC/PyMARL as appropriate.
