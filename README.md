# FALCON

Failure-Aware LLM-Guided Curriculum Optimization for Robust Multi-UAV Decision Making.

FALCON is an outer-loop curriculum optimization framework for multi-UAV reinforcement learning. It records failed rollouts, extracts structured failure summaries, asks an LLM to generate counterfactual curriculum scenarios, validates the generated scenarios through an executable scenario interface, and filters high-value curricula with a dual-boundary difficulty evaluator before MAPPO training.

The LLM is used only to generate candidate training scenarios. It does not output UAV actions, modify the reward function, change the simulator dynamics, or alter the MAPPO algorithm.

## Main Idea

Standard self-play training can overfit to a narrow scenario distribution. Legal LLM-generated scenarios are also not automatically useful: some are too easy, some are not solvable by available policies, and some are redundant. FALCON addresses this by closing the loop between observed policy failures and curriculum generation.

The FALCON pipeline is:

```text
MAPPO rollout
-> trajectory recorder
-> failure analyzer
-> structured failure summary
-> LLM-guided CandidateScenario generation
-> schema and physical constraint validation
-> scenario adapter and YAML materialization
-> environment load/reset check
-> current-policy and historical-best policy evaluation
-> dual-boundary difficulty filtering
-> curriculum pool and scheduler
-> MAPPO training with scenario-config-path
```

## Key Components

### Failure Analyzer

The failure analyzer converts failed rollouts into structured failure summaries. The current implementation tracks failure signals such as coordination failure, target-assignment confusion, initial disadvantage, generalization failure, and overall failure severity.

### CandidateScenario Interface

The LLM outputs structured `CandidateScenario` objects rather than free-form scenario text. Candidate scenarios describe tactical initial conditions, changed factors, target failure modes, and metadata. They are validated before being converted into executable YAML scenarios.

### Constraint Checker and Scenario Adapter

Generated candidates pass schema validation, physical/task constraint checking, YAML materialization, and environment load/reset validation. This separates LLM generation from simulator execution and keeps generated scenarios grounded in the MultiCombat environment.

### Dual-Boundary Difficulty Evaluator

FALCON filters scenarios using both current-policy weakness and historical-best-policy solvability:

```text
W_current(c) <= tau_easy
W_best(c)    >= tau_solve
D(c)         >= tau_diversity
```

This keeps scenarios that are hard for the current policy, still solvable by a stronger historical policy, and sufficiently diverse. Rejection reasons include `too_easy_for_current_policy`, `not_solvable_by_historical_best_policy`, and `insufficient_scenario_diversity`.

### Training Bridge

The MAPPO training entry supports `--scenario-config-path`, allowing generated YAML scenarios to be consumed without changing the MAPPO algorithm core. The original scenario-name based training path remains compatible.

## Repository Structure

```text
algorithms/                         MAPPO/PPO algorithm implementations
envs/                               LAG/JSBSim environment code and scenario configs
falcon/                             FALCON controller, scenario generation, evaluators, schedulers
runner/                             Training/evaluation runners
scripts/falcon/                     FALCON experiment, evaluation, and utility scripts
experiments/falcon_2v2_noweapon/
  configs/                          Lightweight experiment protocol configs
  manifests/                        Lightweight eval/scenario manifests
config.py                           Shared training configuration
test_env.py                         Basic environment smoke entry
```

Large experiment outputs, checkpoints, local LLM weights, generated figures, paper drafts, and analysis artifacts are intentionally excluded from this repository.

## Installation

Create a Python environment:

```bash
conda create -n falcon python=3.8
conda activate falcon
```

Install core dependencies:

```bash
pip install -r requirements.txt
```

Initialize the JSBSim submodule:

```bash
git submodule update --init --recursive
```

For LLM-guided generation, install and run Ollama separately, then make sure the Qwen model is available:

```bash
ollama pull qwen3:8b
```

## Quick Start

Run a lightweight FALCON pilot:

```bash
python scripts/falcon/run_falcon_short_pilot.py ^
  --max-rounds 5 ^
  --train-steps-per-round 256 ^
  --eval-episodes-per-round 2 ^
  --qwen-candidates-per-round 2 ^
  --policy-eval-episodes-per-candidate 2 ^
  --output-dir tests/tmp_falcon_short_pilot
```

Analyze a pilot output directory:

```bash
python scripts/falcon/analyze_falcon_pilot.py ^
  --output-dir tests/tmp_falcon_short_pilot
```

Run a baseline group in smoke mode:

```bash
python scripts/falcon/run_baseline_experiment.py ^
  --group falcon_no_fsn ^
  --seed 0 ^
  --protocol experiments/falcon_2v2_noweapon/configs/experiment_protocol.yaml ^
  --smoke-run
```

Evaluate a checkpoint on the fixed evaluation set:

```bash
python scripts/falcon/evaluate_baseline_on_eval_set.py ^
  --group falcon_no_fsn ^
  --seed 0 ^
  --checkpoint best ^
  --episodes-per-scenario 1 ^
  --opponent-mode fixed_checkpoint
```

## Baselines

The framework supports the following baseline modes:

- `mappo_base`: fixed original 2v2 NoWeapon Selfplay scenario.
- `mappo_random_curriculum`: random legal scenario generation with constraint checking.
- `mappo_qwen_only`: Qwen-generated legal scenarios without full dual-boundary difficulty filtering.
- `falcon_no_fsn`: full FALCON pipeline without FSN.

FSN and OPD are not required for the main FALCON framework in this repository.

## Notes on Reproducibility

This repository contains source code, lightweight configs, and scenario manifests. It does not include large checkpoints, local model weights, full experiment results, paper drafts, or generated analysis assets. To reproduce full experiments, run the provided scripts and regenerate outputs locally.

The original environment is based on the Light Aircraft Game / JSBSim MultiCombat framework with MAPPO training support.

## License

This repository includes and extends code derived from the Light Aircraft Game / CloseAirCombat ecosystem. Please review `LICENSE` and the licenses of upstream dependencies before redistribution or commercial use.

## Citation

If you use this repository, please cite the corresponding FALCON paper when available. Until then, cite this repository and the upstream LAG/JSBSim environment as appropriate.
