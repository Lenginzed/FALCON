"""Validation-set checkpoint selection utilities for FALCON baselines."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .eval_set_evaluator import EvalSetEvaluator

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GROUPS = ("mappo_base", "mappo_random_curriculum", "mappo_qwen_only", "falcon_no_fsn")
DEFAULT_CANDIDATE_ROUNDS = (0, 5, 10, 15, 19)
DEFAULT_VALIDATION_IDS = (
    "base_000",
    "random_000",
    "random_003",
    "hard_random_000",
    "hard_random_003",
    "qwen_pilot_000",
    "qwen_pilot_004",
    "replay_failure_000",
)


class CheckpointSelector:
    """Select a checkpoint using a frozen validation scenario subset."""

    def __init__(
        self,
        results_root: str | Path,
        base_config_path: str | Path,
        validation_manifest_path: str | Path,
        opponent_checkpoint: str | Path,
        opponent_mode: str = "fixed_checkpoint",
    ) -> None:
        self.results_root = _resolve(results_root)
        self.base_config_path = _resolve(base_config_path)
        self.validation_manifest_path = _resolve(validation_manifest_path)
        self.opponent_checkpoint = _resolve(opponent_checkpoint)
        self.opponent_mode = opponent_mode

    def discover_candidates(
        self,
        group: str,
        seed: int,
        candidate_rounds: Optional[Sequence[int]] = None,
        include_latest: bool = True,
    ) -> List[Dict[str, Any]]:
        """Read baseline/FALCON registries and return existing candidate actors."""

        wanted_rounds = set(DEFAULT_CANDIDATE_ROUNDS if candidate_rounds is None else candidate_rounds)
        seed_dir = self.results_root / group / f"seed_{int(seed)}"
        registries = [
            seed_dir / "pilot_run" / "checkpoint_registry.json",
            seed_dir / "pilot_run" / "controller" / "falcon_checkpoint_registry.json",
        ]
        candidates: List[Dict[str, Any]] = []
        warnings: List[str] = []
        for registry_path in registries:
            if not registry_path.exists():
                continue
            registry = _load_json(registry_path)
            for entry in registry.get("checkpoints") or []:
                if not isinstance(entry, MappingABC):
                    continue
                round_id = _safe_int(entry.get("round_id"))
                if round_id is None or round_id not in wanted_rounds:
                    continue
                path = _resolve(entry.get("checkpoint_path"))
                if not path.exists():
                    warnings.append(f"Checkpoint listed but missing: {path}")
                    continue
                candidates.append(
                    {
                        "group": group,
                        "seed": int(seed),
                        "round_id": round_id,
                        "candidate_label": f"round_{round_id:03d}",
                        "candidate_checkpoint": str(path),
                        "registry_path": str(registry_path),
                        "registry_role": entry.get("role"),
                        "registry_eval_win_rate": entry.get("eval_win_rate"),
                    }
                )
            if include_latest:
                latest_path = registry.get("latest_checkpoint") or registry.get("current_checkpoint")
                latest_round = _round_for_checkpoint(registry, latest_path)
                path = _resolve(latest_path)
                if latest_path and path.exists():
                    candidates.append(
                        {
                            "group": group,
                            "seed": int(seed),
                            "round_id": latest_round,
                            "candidate_label": "latest",
                            "candidate_checkpoint": str(path),
                            "registry_path": str(registry_path),
                            "registry_role": "latest_checkpoint",
                            "registry_eval_win_rate": None,
                        }
                    )
        deduped = _dedupe_candidates(candidates)
        for row in deduped:
            row["warnings"] = list(warnings)
        return deduped

    def select_for_group_seed(
        self,
        group: str,
        seed: int,
        output_dir: str | Path,
        candidate_rounds: Optional[Sequence[int]] = None,
        episodes_per_scenario: int = 1,
        include_latest: bool = True,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate candidate checkpoints on validation split and select best."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        candidates = self.discover_candidates(group, seed, candidate_rounds, include_latest=include_latest)
        warnings: List[str] = []
        rows: List[Dict[str, Any]] = []
        evaluator = EvalSetEvaluator(
            self.validation_manifest_path,
            {"base_config_path": str(self.base_config_path)},
        )
        for candidate in candidates:
            checkpoint = Path(candidate["candidate_checkpoint"])
            round_label = candidate["candidate_label"]
            round_dir = output_dir / round_label
            summary_path = round_dir / "eval_set_summary.json"
            if summary_path.exists() and not force:
                eval_summary = _load_json(summary_path)
            else:
                eval_summary = evaluator.evaluate_checkpoint(
                    checkpoint,
                    episodes_per_scenario=int(episodes_per_scenario),
                    seed=int(seed),
                    group=group,
                    checkpoint_role=round_label,
                    opponent_mode=self.opponent_mode,
                    opponent_checkpoint=self.opponent_checkpoint,
                )
                EvalSetEvaluator.save(eval_summary, summary_path)
            aggregate = dict(eval_summary.get("aggregate_result") or {})
            row = {
                "group": group,
                "seed": int(seed),
                "candidate_checkpoint": str(checkpoint),
                "round_id": candidate.get("round_id"),
                "candidate_label": round_label,
                "validation_win_rate": _safe_float(aggregate.get("final_win_rate"), 0.0),
                "validation_mean_return": _safe_float(aggregate.get("final_mean_return"), 0.0),
                "num_validation_scenarios": eval_summary.get("num_scenarios_evaluated"),
                "failure_stage": eval_summary.get("failure_stage"),
                "same_actor": eval_summary.get("same_actor"),
                "same_checkpoint": eval_summary.get("same_checkpoint"),
                "opponent_mode": eval_summary.get("opponent_mode"),
                "opponent_checkpoint": eval_summary.get("opponent_checkpoint"),
                "selected": False,
                "selection_reason": None,
                "eval_summary_path": str(summary_path),
                "warnings": "; ".join(str(item) for item in eval_summary.get("warnings") or []),
            }
            rows.append(row)
            if row["failure_stage"]:
                warnings.append(f"{group} seed {seed} {round_label} validation failed: {row['failure_stage']}")
        selected = _select_best_row(rows)
        if selected is not None:
            selected["selected"] = True
            selected["selection_reason"] = (
                "highest validation_win_rate; validation_mean_return used as tie-breaker; "
                "earliest round used only for remaining exact ties"
            )
        selected_path = output_dir / "validation_selected_checkpoint.json"
        csv_path = output_dir / "checkpoint_validation_results.csv"
        _write_csv(csv_path, rows)
        selection_summary = {
            "schema_version": "falcon.validation_checkpoint_selection.v1",
            "created_at": _timestamp(),
            "group": group,
            "seed": int(seed),
            "candidate_rounds": list(candidate_rounds or DEFAULT_CANDIDATE_ROUNDS),
            "include_latest": bool(include_latest),
            "num_candidates_found": len(candidates),
            "num_candidates_evaluated": len(rows),
            "episodes_per_validation_scenario": int(episodes_per_scenario),
            "validation_manifest_path": str(self.validation_manifest_path),
            "opponent_mode": self.opponent_mode,
            "opponent_checkpoint": str(self.opponent_checkpoint),
            "selected_checkpoint": selected.get("candidate_checkpoint") if selected else None,
            "selected_round_id": selected.get("round_id") if selected else None,
            "selected_candidate_label": selected.get("candidate_label") if selected else None,
            "validation_win_rate": selected.get("validation_win_rate") if selected else None,
            "validation_mean_return": selected.get("validation_mean_return") if selected else None,
            "validation_results_csv": str(csv_path),
            "candidate_results": rows,
            "failure_stage": None if selected else "no_valid_checkpoint_candidate",
            "warnings": sorted(set(warnings)),
        }
        _write_json(selected_path, selection_summary)
        return selection_summary


class FailureBalancedCheckpointSelector:
    """Score existing checkpoints on an independent failure-balanced proxy set."""

    DEFAULT_WEIGHTS = {
        "initial_disadvantage_win": 0.30,
        "coordination_win": 0.25,
        "target_assignment_win": 0.25,
        "overall_validation_win": 0.20,
    }

    def __init__(self, weights: Optional[Mapping[str, Any]] = None) -> None:
        self.weights = dict(self.DEFAULT_WEIGHTS)
        self.weights.update(dict(weights or {}))

    def score(self, candidate: Mapping[str, Any]) -> Dict[str, float]:
        weighted_score = sum(
            float(self.weights[key]) * _safe_float(candidate.get(key), 0.0)
            for key in self.DEFAULT_WEIGHTS
        )
        worst_group_score = min(
            _safe_float(candidate.get("initial_disadvantage_win"), 0.0),
            _safe_float(candidate.get("coordination_win"), 0.0),
            _safe_float(candidate.get("target_assignment_win"), 0.0),
        )
        return {
            "failure_balanced_score": round(weighted_score, 6),
            "worst_group_score": round(worst_group_score, 6),
        }

    def select(
        self,
        candidates: Sequence[Mapping[str, Any]],
        mode: str = "weighted",
    ) -> Optional[Dict[str, Any]]:
        scored = []
        for candidate in candidates:
            row = dict(candidate)
            row.update(self.score(row))
            scored.append(row)
        if not scored:
            return None
        if mode == "weighted":
            key = lambda row: (
                row["failure_balanced_score"],
                _safe_float(row.get("accepted_scene_win_rate"), 0.0),
                _safe_float(row.get("proxy_mean_return"), float("-inf")),
                _safe_int(row.get("source_round")) or 0,
            )
        elif mode == "worst_group":
            key = lambda row: (
                row["worst_group_score"],
                _safe_float(row.get("overall_validation_win"), 0.0),
                _safe_float(row.get("proxy_mean_return"), float("-inf")),
                _safe_int(row.get("source_round")) or 0,
            )
        else:
            raise ValueError(f"Unsupported failure-balanced selector mode: {mode}")
        return max(scored, key=key)


def create_eval_split(
    source_manifest_path: str | Path,
    split_path: str | Path,
    validation_ids: Sequence[str] = DEFAULT_VALIDATION_IDS,
    force: bool = False,
) -> Dict[str, Any]:
    """Create a fixed validation/test split and two subset manifests."""

    source_manifest_path = _resolve(source_manifest_path)
    split_path = _resolve(split_path)
    validation_manifest_path = split_path.with_name(f"{split_path.stem}_validation.json")
    test_manifest_path = split_path.with_name(f"{split_path.stem}_test.json")
    if split_path.exists() and validation_manifest_path.exists() and test_manifest_path.exists() and not force:
        return _load_json(split_path)
    manifest = _load_json(source_manifest_path)
    scenarios = list(manifest.get("scenarios") or [])
    by_id = {str(item.get("scenario_id")): item for item in scenarios if isinstance(item, MappingABC)}
    validation_set = [by_id[item] for item in validation_ids if item in by_id]
    validation_id_set = {str(item.get("scenario_id")) for item in validation_set}
    test_set = [item for item in scenarios if str(item.get("scenario_id")) not in validation_id_set]
    overlap = validation_id_set.intersection(str(item.get("scenario_id")) for item in test_set)
    missing_validation = [item for item in validation_ids if item not in by_id]
    warnings: List[str] = []
    if missing_validation:
        warnings.append(f"Requested validation scenario ids not found: {missing_validation}")
    if overlap:
        warnings.append(f"Split overlap detected: {sorted(overlap)}")
    if _group_counts(validation_set).get("base") and not _group_counts(test_set).get("base"):
        warnings.append(
            "Only one base scenario exists; it is assigned to validation, so held-out test has no base scenario."
        )
    validation_manifest = _subset_manifest(
        manifest,
        validation_set,
        parent_split=str(split_path),
        split_name="validation",
    )
    test_manifest = _subset_manifest(
        manifest,
        test_set,
        parent_split=str(split_path),
        split_name="test",
    )
    split = {
        "schema_version": "falcon.eval_split.v1",
        "split_id": split_path.stem,
        "created_at": _timestamp(),
        "source_manifest": str(source_manifest_path),
        "fixed_across_groups_and_seeds": True,
        "validation": {
            "manifest_path": str(validation_manifest_path),
            "scenario_count": len(validation_set),
            "scenario_ids": [str(item.get("scenario_id")) for item in validation_set],
            "scenario_group_counts": _group_counts(validation_set),
        },
        "test": {
            "manifest_path": str(test_manifest_path),
            "scenario_count": len(test_set),
            "scenario_ids": [str(item.get("scenario_id")) for item in test_set],
            "scenario_group_counts": _group_counts(test_set),
        },
        "overlap_scenario_ids": sorted(overlap),
        "warnings": warnings,
    }
    _write_json(validation_manifest_path, validation_manifest)
    _write_json(test_manifest_path, test_manifest)
    _write_json(split_path, split)
    return split


def write_per_scenario_csv(summary: Mapping[str, Any], output_path: str | Path) -> None:
    rows = []
    for row in summary.get("per_scenario_results") or []:
        if not isinstance(row, MappingABC):
            continue
        rows.append(
            {
                "scenario_id": row.get("scenario_id"),
                "scenario_group": row.get("scenario_group"),
                "win_rate": row.get("win_rate"),
                "mean_return": row.get("mean_return"),
                "std_return": row.get("std_return"),
                "mean_episode_length": row.get("mean_episode_length"),
                "failure_stage": row.get("failure_stage"),
            }
        )
    _write_csv(output_path, rows)


def _subset_manifest(
    source_manifest: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    parent_split: str,
    split_name: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "falcon.eval_scenario_manifest.v1",
        "parent_manifest": source_manifest.get("source_manifest") or source_manifest.get("manifest_path"),
        "parent_split": parent_split,
        "split_name": split_name,
        "scenario_count": len(scenarios),
        "scenario_group_counts": _group_counts(scenarios),
        "scenarios": list(scenarios),
        "warnings": list(source_manifest.get("warnings") or []),
    }


def _group_counts(scenarios: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter(str(item.get("scenario_group") or "unknown") for item in scenarios)
    return dict(sorted(counts.items()))


def _round_for_checkpoint(registry: Mapping[str, Any], checkpoint_path: Any) -> Optional[int]:
    if not checkpoint_path:
        return None
    target = str(_resolve(checkpoint_path))
    for entry in registry.get("checkpoints") or []:
        if not isinstance(entry, MappingABC):
            continue
        try:
            if str(_resolve(entry.get("checkpoint_path"))) == target:
                return _safe_int(entry.get("round_id"))
        except (TypeError, ValueError):
            continue
    return None


def _dedupe_candidates(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    best_by_round: Dict[int, Dict[str, Any]] = {}
    latest: Optional[Dict[str, Any]] = None
    priority = {"round_checkpoint": 0, None: 1, "initial_current": 2, "latest_checkpoint": 3, "evaluated_current": 4}
    for candidate in candidates:
        row = dict(candidate)
        label = str(row.get("candidate_label") or "")
        if label == "latest":
            latest = row
            continue
        round_id = _safe_int(row.get("round_id"))
        if round_id is None:
            continue
        current = best_by_round.get(round_id)
        if current is None or priority.get(row.get("registry_role"), 5) < priority.get(current.get("registry_role"), 5):
            best_by_round[round_id] = row
    rows = [best_by_round[key] for key in sorted(best_by_round)]
    if latest:
        latest_path = str(latest.get("candidate_checkpoint"))
        if not any(str(row.get("candidate_checkpoint")) == latest_path for row in rows):
            rows.append(latest)
    return rows


def _select_best_row(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = [row for row in rows if not row.get("failure_stage")]
    if not valid:
        return None
    return max(
        valid,
        key=lambda row: (
            _safe_float(row.get("validation_win_rate"), 0.0),
            _safe_float(row.get("validation_mean_return"), 0.0),
            -(_safe_int(row.get("round_id")) if _safe_int(row.get("round_id")) is not None else 10**9),
        ),
    )


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _resolve(value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else ROOT_DIR / path


def _load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(data) if isinstance(data, MappingABC) else {}


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, sort_keys=True)


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
