# FALCON 2v2 NoWeapon Baseline Protocol

This directory freezes the preparation-stage protocol for comparing:

- `mappo_base`
- `mappo_random_curriculum`
- `mappo_qwen_only`
- `falcon_no_fsn`

The current protocol intentionally enables only dry runs and bounded smoke
runs. It does not start formal multi-seed experiments.

Prepare or refresh the fixed evaluation manifest:

```powershell
python scripts\falcon\prepare_baseline_eval_scenarios.py
```

Validate a group without training:

```powershell
python scripts\falcon\run_baseline_experiment.py --group falcon_no_fsn --seed 0 --dry-run
```

Run a bounded smoke:

```powershell
python scripts\falcon\run_baseline_experiment.py --group falcon_no_fsn --seed 0 --smoke-run
```

