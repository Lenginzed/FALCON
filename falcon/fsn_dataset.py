"""Dataset construction and audit helpers for offline FSN distillation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .trajectory_recorder import SCENARIO_VECTOR_KEYS

FSN_DATASET_SCHEMA_VERSION = "falcon.fsn_dataset.v1"
FSN_SAMPLE_SCHEMA_VERSION = "falcon.fsn_sample.v1"

FAILURE_KEYS = (
    "coordination_failure",
    "target_assignment_confusion",
    "initial_disadvantage",
    "generalization_failure",
    "failure_severity",
)

POLICY_KEYS = ("W_current", "W_best", "R_current", "R_best")

PROXY_FEATURE_KEYS = (
    "initial_disadvantage_proxy",
    "distance_disadvantage",
    "heading_disadvantage",
    "altitude_disadvantage",
    "velocity_disadvantage",
    "pool_novelty_score",
    "policy_performance_drop_proxy",
)

LABELS = (
    "accepted",
    "rejected_too_easy",
    "rejected_not_solvable",
    "rejected_low_diversity",
    "invalid",
)

DEFAULT_SOURCE_ROOTS = (
    "experiments/falcon_2v2_noweapon/results",
    "experiments/falcon_2v2_noweapon/results_longer_budget",
    "experiments/falcon_2v2_noweapon/results_coverage_aware_longer_budget",
    "experiments/falcon_2v2_noweapon/results_stability_aware_longer_budget",
    "tests",
)


class FSNDatasetBuilder:
    """Build compact FSN samples from persistent curriculum pool artifacts."""

    def __init__(
        self,
        workspace_root: str | Path,
        source_roots: Optional[Sequence[str | Path]] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        roots = source_roots or DEFAULT_SOURCE_ROOTS
        self.source_roots = [
            self._resolve(root) for root in roots if self._resolve(root).exists()
        ]
        self.warnings: List[str] = []

    def discover_pool_files(self) -> List[Path]:
        """Prefer final pools and standalone smoke pools, avoiding round snapshots."""

        files: set[Path] = set()
        for root in self.source_roots:
            files.update(root.rglob("falcon_curriculum_pool_final.json"))
            for pattern in (
                "falcon_curriculum_pool.json",
                "multicheckpoint_curriculum_pool.json",
            ):
                files.update(root.rglob(pattern))
        return sorted(path.resolve() for path in files)

    def build(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        source_files = self.discover_pool_files()
        for pool_path in source_files:
            payload = _load_json(pool_path)
            for item in payload.get("items") or []:
                if not isinstance(item, MappingABC):
                    continue
                samples.append(self._sample_from_pool_item(pool_path, item))

        sample_ids: set[str] = set()
        unique_samples: List[Dict[str, Any]] = []
        for sample in samples:
            sample_id = str(sample["sample_id"])
            if sample_id in sample_ids:
                continue
            sample_ids.add(sample_id)
            unique_samples.append(sample)
        samples = unique_samples

        split_counts = Counter(str(item.get("split")) for item in samples)
        seeds = {item.get("seed") for item in samples if item.get("seed") is not None}
        split_strategy = "seed_group_split"
        if not {0, 1, 2}.intersection(seeds) or 3 not in seeds or 4 not in seeds:
            split_strategy = "deterministic_70_15_15"
            self.warnings.append(
                "Seed group split was unavailable; deterministic 70/15/15 split was used."
            )
            for sample in samples:
                sample["split"] = _hash_split(sample["sample_id"])
            split_counts = Counter(str(item.get("split")) for item in samples)

        label_counts = Counter(str(item.get("label")) for item in samples)
        if label_counts.get("invalid", 0) < 20:
            self.warnings.append(
                "Invalid pool items are scarce. FSN cannot reliably learn physical "
                "validity; constraint_checker remains mandatory."
            )
        summary = {
            "schema_version": FSN_DATASET_SCHEMA_VERSION,
            "total_samples": len(samples),
            "source_file_count": len(source_files),
            "source_files": [self._relative(path) for path in source_files],
            "label_counts": _ordered_counts(label_counts, LABELS),
            "accepted_count": label_counts.get("accepted", 0),
            "rejected_count": sum(
                label_counts.get(label, 0)
                for label in LABELS
                if label.startswith("rejected_")
            ),
            "invalid_count": label_counts.get("invalid", 0),
            "split_strategy": split_strategy,
            "split_counts": dict(sorted(split_counts.items())),
            "seed_counts": dict(
                sorted(
                    Counter(str(item.get("seed")) for item in samples).items()
                )
            ),
            "generator_type_counts": dict(
                sorted(Counter(item.get("generator_type") for item in samples).items())
            ),
            "warnings": sorted(set(self.warnings)),
        }
        return samples, summary

    def write(
        self,
        samples: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
        dataset_path: str | Path,
        summary_path: str | Path,
        feature_stats_path: str | Path,
    ) -> None:
        dataset_path = Path(dataset_path)
        summary_path = Path(summary_path)
        feature_stats_path = Path(feature_stats_path)
        for path in (dataset_path, summary_path, feature_stats_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        with dataset_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(dict(sample), sort_keys=True) + "\n")
        summary_path.write_text(
            json.dumps(dict(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_feature_stats(samples, feature_stats_path)

    def _sample_from_pool_item(
        self,
        pool_path: Path,
        item: Mapping[str, Any],
    ) -> Dict[str, Any]:
        candidate = dict(item.get("candidate_scenario") or {})
        difficulty = dict(item.get("difficulty_result") or {})
        constraint = dict(item.get("constraint_result") or {})
        policy = dict(item.get("policy_eval_result") or {})
        failure = dict(item.get("failure_vector") or {})
        failure_scores = dict(failure.get("failure_scores") or {})
        current_eval, best_eval = _policy_evals(policy, difficulty)
        reasons = list(
            difficulty.get("rejection_reasons")
            or (item.get("metadata") or {}).get("rejection_reasons")
            or []
        )
        constraint_valid = bool(constraint.get("is_valid", False))
        accepted = bool(
            item.get(
                "accepted_into_curriculum_pool",
                difficulty.get("accepted_into_curriculum_pool", False),
            )
        )
        label = _label(accepted, constraint_valid, reasons)
        seed = _seed_from_path(pool_path)
        round_id = _int_or_none(item.get("source_round"))
        source_run = self._source_run(pool_path)
        scenario_id = str(
            item.get("scenario_id")
            or candidate.get("scenario_id")
            or item.get("pool_item_id")
            or "unknown"
        )
        sample_id = _sample_id(source_run, seed, round_id, scenario_id)
        value_score = _nullable_float(
            item.get("final_value_score", difficulty.get("final_value_score"))
        )
        return {
            "schema_version": FSN_SAMPLE_SCHEMA_VERSION,
            "sample_id": sample_id,
            "source_run": source_run,
            "source_pool_path": self._relative(pool_path),
            "seed": seed,
            "round_id": round_id,
            "scenario_id": scenario_id,
            "source_failure_id": item.get("source_failure_id")
            or candidate.get("source_failure_id"),
            "failure_vector": {
                key: _nullable_float(
                    failure_scores.get(
                        key,
                        failure.get(key)
                        if key != "failure_severity"
                        else failure.get("failure_severity"),
                    )
                )
                for key in FAILURE_KEYS
            },
            "primary_failure_modes": list(
                failure.get("primary_failure_modes")
                or candidate.get("target_failure_modes")
                or []
            ),
            "secondary_failure_modes": list(
                failure.get("secondary_failure_modes") or []
            ),
            "candidate_scenario_vector": {
                key: _nullable_float(
                    (item.get("scenario_vector") or candidate.get("scenario_vector") or {}).get(key)
                )
                for key in SCENARIO_VECTOR_KEYS
            },
            "changed_factors": list(candidate.get("changed_factors") or []),
            "target_failure_modes": list(candidate.get("target_failure_modes") or []),
            "constraint_valid": constraint_valid,
            "policy_eval": {
                "W_current": _nullable_float(current_eval.get("win_rate")),
                "W_best": _nullable_float(best_eval.get("win_rate")),
                "R_current": _nullable_float(current_eval.get("mean_return")),
                "R_best": _nullable_float(best_eval.get("mean_return")),
            },
            "difficulty": {
                "final_value_score": value_score,
                "learning_potential": _nullable_float(
                    difficulty.get("learning_potential")
                ),
                "diversity_score": _nullable_float(
                    difficulty.get("scenario_diversity")
                ),
                "rejection_reasons": sorted(set(str(reason) for reason in reasons)),
            },
            "label": label,
            "generator_type": _generator_type(
                candidate.get("generator_type") or item.get("source")
            ),
            "scenario_yaml_path": item.get("scenario_yaml_path")
            or candidate.get("scenario_yaml_path")
            or candidate.get("yaml_path"),
            "sample_weight": _sample_weight(label, value_score),
            "split": _seed_split(seed),
        }

    def _source_run(self, pool_path: Path) -> str:
        try:
            relative = pool_path.parent.relative_to(self.workspace_root)
        except ValueError:
            relative = pool_path.parent
        return str(relative).replace("\\", "/")

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workspace_root)).replace("\\", "/")
        except ValueError:
            return str(path)

    def _resolve(self, path: str | Path) -> Path:
        value = Path(path)
        return value.resolve() if value.is_absolute() else (self.workspace_root / value).resolve()


class FSNStage2DatasetBuilder:
    """Deduplicate stage-1 samples, group splits by scenario hash, and add negatives."""

    SCENARIO_SCALES = {
        "team_center_distance": 10000.0,
        "own_formation_spread": 3000.0,
        "opponent_formation_spread": 3000.0,
        "altitude_difference": 2000.0,
        "velocity_difference": 150.0,
        "heading_difference": math.pi,
        "approximate_aspect_angle": math.pi,
        "own_center_x": 10000.0,
        "own_center_y": 10000.0,
        "own_center_z": 8000.0,
        "opponent_center_x": 10000.0,
        "opponent_center_y": 10000.0,
        "opponent_center_z": 8000.0,
    }

    INVALID_REASONS = (
        "altitude_below_minimum",
        "altitude_above_maximum",
        "velocity_above_maximum",
        "team_center_distance_below_minimum",
        "team_center_distance_above_maximum",
        "own_formation_spread_below_minimum",
        "opponent_formation_spread_above_maximum",
        "heading_difference_out_of_range",
        "aspect_angle_out_of_range",
        "minimum_safety_separation_violation",
    )

    def __init__(
        self,
        approximate_hash_resolution: float = 0.01,
        invalid_ratio_to_accepted: float = 0.40,
        seed: int = 17,
    ) -> None:
        self.hash_resolution = max(float(approximate_hash_resolution), 1e-6)
        self.invalid_ratio = min(max(float(invalid_ratio_to_accepted), 0.0), 1.0)
        self.seed = int(seed)

    def build(
        self, stage1_samples: Sequence[Mapping[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        enriched = [self._enrich(dict(sample)) for sample in stage1_samples]
        scenario_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sample in enriched:
            scenario_hash = self.scenario_hash(
                sample.get("candidate_scenario_vector") or {}
            )
            sample["scenario_group_hash"] = scenario_hash
            scenario_groups[scenario_hash].append(sample)

        deduplicated: List[Dict[str, Any]] = []
        semantic_duplicate_count = 0
        for scenario_hash, group in scenario_groups.items():
            seen: set[str] = set()
            for sample in group:
                semantic_hash = self._semantic_hash(sample)
                if semantic_hash in seen:
                    semantic_duplicate_count += 1
                    continue
                seen.add(semantic_hash)
                sample["semantic_dedup_hash"] = semantic_hash
                deduplicated.append(sample)

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sample in deduplicated:
            grouped[str(sample["scenario_group_hash"])].append(sample)

        split_strategy = "seed_grouped_by_scenario_hash"
        assignments, conflicts = self._assign_seed_grouped_splits(grouped)
        split_group_counts = Counter(assignments.values())
        if (
            split_group_counts.get("val", 0) < 5
            or split_group_counts.get("test", 0) < 5
        ):
            split_strategy = "hash_grouped_70_15_15"
            assignments = {
                scenario_hash: _hash_split(scenario_hash)
                for scenario_hash in grouped
            }

        for scenario_hash, group in grouped.items():
            for sample in group:
                sample["split"] = assignments[scenario_hash]

        accepted = [
            sample for sample in deduplicated if sample.get("label") == "accepted"
        ]
        synthetic_count = int(round(len(accepted) * self.invalid_ratio))
        synthetic = self._synthetic_invalid_samples(accepted, synthetic_count)
        real_hash_splits = {
            str(sample.get("scenario_group_hash")): str(sample.get("split"))
            for sample in deduplicated
        }
        synthetic_hash_splits: Dict[str, str] = {}
        for sample in synthetic:
            scenario_hash = str(sample.get("scenario_group_hash"))
            split = real_hash_splits.get(scenario_hash)
            if split is None:
                split = synthetic_hash_splits.setdefault(
                    scenario_hash, _hash_split(scenario_hash)
                )
            sample["split"] = split
        all_samples = deduplicated + synthetic

        split_hashes: Dict[str, set[str]] = defaultdict(set)
        for sample in all_samples:
            split_hashes[str(sample.get("split"))].add(
                str(sample.get("scenario_group_hash"))
            )
        overlap = {
            "train_val": sorted(split_hashes["train"] & split_hashes["val"]),
            "train_test": sorted(split_hashes["train"] & split_hashes["test"]),
            "val_test": sorted(split_hashes["val"] & split_hashes["test"]),
        }
        leakage_count = sum(len(values) for values in overlap.values())
        summary = {
            "schema_version": "falcon.fsn_stage2_dataset.v1",
            "stage1_sample_count": len(stage1_samples),
            "stage1_scenario_hash_group_count": len(scenario_groups),
            "semantic_duplicate_samples_removed": semantic_duplicate_count,
            "deduplicated_real_sample_count": len(deduplicated),
            "accepted_real_sample_count": len(accepted),
            "synthetic_invalid_count": len(synthetic),
            "synthetic_invalid_to_accepted_ratio": round(
                len(synthetic) / max(len(accepted), 1), 6
            ),
            "total_stage2_samples": len(all_samples),
            "split_strategy": split_strategy,
            "split_counts": dict(
                sorted(Counter(str(item.get("split")) for item in all_samples).items())
            ),
            "label_counts": dict(
                sorted(Counter(str(item.get("label")) for item in all_samples).items())
            ),
            "scenario_hash_group_count": len(
                {str(item.get("scenario_group_hash")) for item in all_samples}
            ),
            "cross_seed_scenario_groups": conflicts,
            "cross_split_scenario_hash_overlap_count": leakage_count,
            "cross_split_leakage_detected": leakage_count > 0,
            "warnings": [
                "Synthetic invalid samples are used only for label/constraint losses.",
                "Stage2 metrics remain offline and do not establish policy improvement.",
            ],
        }
        split_audit = {
            "schema_version": "falcon.fsn_stage2_split_audit.v1",
            "split_strategy": split_strategy,
            "hash_resolution": self.hash_resolution,
            "scenario_hash_group_count": summary["scenario_hash_group_count"],
            "split_counts": summary["split_counts"],
            "split_hash_counts": {
                split: len(values) for split, values in sorted(split_hashes.items())
            },
            "cross_seed_scenario_groups": conflicts,
            "cross_split_overlap": overlap,
            "cross_split_scenario_hash_overlap_count": leakage_count,
            "cross_split_leakage_detected": leakage_count > 0,
        }
        return all_samples, synthetic, summary, split_audit

    def scenario_hash(self, vector: Mapping[str, Any]) -> str:
        normalized = []
        for key in SCENARIO_VECTOR_KEYS:
            value = _nullable_float(vector.get(key))
            if value is None:
                normalized.append(None)
                continue
            scaled = value / max(float(self.SCENARIO_SCALES[key]), 1e-8)
            normalized.append(round(scaled / self.hash_resolution))
        return hashlib.sha256(json.dumps(normalized).encode("utf-8")).hexdigest()[:20]

    def write(
        self,
        samples: Sequence[Mapping[str, Any]],
        synthetic: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
        split_audit: Mapping[str, Any],
        output_dir: str | Path,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "failure_to_scenario_dataset_dedup.jsonl", samples)
        _write_jsonl(output_dir / "synthetic_invalid_samples.jsonl", synthetic)
        (output_dir / "failure_to_scenario_dataset_dedup_summary.json").write_text(
            json.dumps(dict(summary), indent=2, sort_keys=True), encoding="utf-8"
        )
        (output_dir / "failure_to_scenario_split_audit.json").write_text(
            json.dumps(dict(split_audit), indent=2, sort_keys=True), encoding="utf-8"
        )
        synthetic_summary = {
            "schema_version": "falcon.fsn_synthetic_invalid_summary.v1",
            "total_samples": len(synthetic),
            "invalid_reason_counts": dict(
                sorted(
                    Counter(str(item.get("invalid_reason")) for item in synthetic).items()
                )
            ),
            "split_counts": dict(
                sorted(Counter(str(item.get("split")) for item in synthetic).items())
            ),
            "participates_in_scenario_regression": False,
            "participates_in_label_classification": True,
            "participates_in_constraint_head": True,
        }
        (output_dir / "synthetic_invalid_summary.json").write_text(
            json.dumps(synthetic_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_feature_stats(
            samples, output_dir / "failure_to_scenario_feature_stats.csv"
        )

    def _enrich(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        vector = dict(sample.get("candidate_scenario_vector") or {})
        policy = dict(sample.get("policy_eval") or {})
        difficulty = dict(sample.get("difficulty") or {})
        team_distance = _nullable_float(vector.get("team_center_distance"))
        own_spread = _nullable_float(vector.get("own_formation_spread"))
        opponent_spread = _nullable_float(vector.get("opponent_formation_spread"))
        altitude = abs(_nullable_float(vector.get("altitude_difference")) or 0.0)
        velocity = abs(_nullable_float(vector.get("velocity_difference")) or 0.0)
        heading = abs(_nullable_float(vector.get("heading_difference")) or 0.0)
        aspect = abs(_nullable_float(vector.get("approximate_aspect_angle")) or 0.0)
        distance_penalty = _clip01(
            abs((team_distance or 12000.0) - 12000.0) / 6000.0
        )
        formation_penalty = _clip01(
            abs((own_spread or 3000.0) - 3000.0) / 3000.0
            + max((opponent_spread or 3000.0) - (own_spread or 3000.0), 0.0)
            / 5000.0
        )
        altitude_penalty = _clip01(altitude / 2500.0)
        velocity_penalty = _clip01(velocity / 80.0)
        heading_penalty = _clip01(
            min(abs(heading - math.pi), abs(aspect - math.pi)) / math.pi
        )
        initial_proxy = _clip01(
            0.25 * distance_penalty
            + 0.20 * formation_penalty
            + 0.20 * altitude_penalty
            + 0.15 * velocity_penalty
            + 0.20 * heading_penalty
        )
        current = _nullable_float(policy.get("W_current"))
        best = _nullable_float(policy.get("W_best"))
        performance_drop = _clip01((best or 0.0) - (current or 0.0))
        sample["proxy_features"] = {
            "initial_disadvantage_proxy": initial_proxy,
            "distance_disadvantage": distance_penalty,
            "heading_disadvantage": heading_penalty,
            "altitude_disadvantage": altitude_penalty,
            "velocity_disadvantage": velocity_penalty,
            "pool_novelty_score": _nullable_float(
                difficulty.get("diversity_score")
            )
            or 0.0,
            "policy_performance_drop_proxy": performance_drop,
        }
        return sample

    def _assign_seed_grouped_splits(
        self, grouped: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> Tuple[Dict[str, str], int]:
        assignments: Dict[str, str] = {}
        conflicts = 0
        for scenario_hash, group in grouped.items():
            desired = {_seed_split(_int_or_none(item.get("seed"))) for item in group}
            if len(desired) > 1:
                conflicts += 1
            if "train" in desired:
                assignments[scenario_hash] = "train"
            elif "val" in desired:
                assignments[scenario_hash] = "val"
            else:
                assignments[scenario_hash] = "test"
        return assignments, conflicts

    def _semantic_hash(self, sample: Mapping[str, Any]) -> str:
        failure = [
            round(_nullable_float((sample.get("failure_vector") or {}).get(key)) or 0.0, 4)
            for key in FAILURE_KEYS
        ]
        payload = {
            "scenario": sample.get("scenario_group_hash"),
            "failure": failure,
            "label": sample.get("label"),
            "changed": sorted(str(value) for value in sample.get("changed_factors") or []),
            "policy": [
                round(_nullable_float((sample.get("policy_eval") or {}).get(key)) or 0.0, 3)
                for key in POLICY_KEYS
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]

    def _synthetic_invalid_samples(
        self, accepted: Sequence[Mapping[str, Any]], count: int
    ) -> List[Dict[str, Any]]:
        if not accepted or count <= 0:
            return []
        synthetic: List[Dict[str, Any]] = []
        for index in range(count):
            source = dict(accepted[index % len(accepted)])
            reason = self.INVALID_REASONS[index % len(self.INVALID_REASONS)]
            vector = dict(source.get("candidate_scenario_vector") or {})
            self._apply_invalid_mutation(vector, reason, index)
            sample = json.loads(json.dumps(source))
            sample["sample_id"] = f"fsn_stage2_invalid_{index:06d}"
            sample["scenario_id"] = f"synthetic_invalid_{index:06d}"
            sample["candidate_scenario_vector"] = vector
            sample["scenario_group_hash"] = self.scenario_hash(vector)
            sample["semantic_dedup_hash"] = sample["sample_id"]
            sample["label"] = "invalid"
            sample["constraint_valid"] = False
            sample["invalid_reason"] = reason
            sample["synthetic"] = True
            sample["generator_type"] = "synthetic_invalid"
            sample["scenario_yaml_path"] = None
            sample["sample_weight"] = 0.1
            sample["changed_factors"] = [
                _invalid_factor(reason)
            ]
            difficulty = dict(sample.get("difficulty") or {})
            difficulty["rejection_reasons"] = [reason]
            difficulty["final_value_score"] = 0.0
            sample["difficulty"] = difficulty
            sample = self._enrich(sample)
            synthetic.append(sample)
        return synthetic

    @staticmethod
    def _apply_invalid_mutation(
        vector: Dict[str, Any], reason: str, index: int
    ) -> None:
        offset = float(index % 7)
        if reason == "altitude_below_minimum":
            vector["own_center_z"] = 1000.0 - offset
            vector["opponent_center_z"] = 1000.0 - offset
        elif reason == "altitude_above_maximum":
            vector["own_center_z"] = 11000.0 + offset
            vector["opponent_center_z"] = 11000.0 + offset
        elif reason == "velocity_above_maximum":
            vector["velocity_difference"] = 120.0 + offset
        elif reason == "team_center_distance_below_minimum":
            vector["team_center_distance"] = 200.0 + offset
        elif reason == "team_center_distance_above_maximum":
            vector["team_center_distance"] = 26000.0 + offset
        elif reason == "own_formation_spread_below_minimum":
            vector["own_formation_spread"] = 100.0 + offset
        elif reason == "opponent_formation_spread_above_maximum":
            vector["opponent_formation_spread"] = 10000.0 + offset
        elif reason == "heading_difference_out_of_range":
            vector["heading_difference"] = 2.0 * math.pi + 0.5 + offset * 0.01
        elif reason == "aspect_angle_out_of_range":
            vector["approximate_aspect_angle"] = (
                2.0 * math.pi + 0.5 + offset * 0.01
            )
        elif reason == "minimum_safety_separation_violation":
            vector["own_formation_spread"] = 100.0 + offset
            vector["team_center_distance"] = 200.0 + offset


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze_samples(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    label_counts = Counter(str(item.get("label")) for item in samples)
    rejection_counts: Counter[str] = Counter()
    failure_mode_counts: Counter[str] = Counter()
    changed_factor_counts: Counter[str] = Counter()
    seed_counts: Counter[str] = Counter()
    run_counts: Counter[str] = Counter()
    vector_signatures: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    failure_values: Dict[str, List[float]] = defaultdict(list)
    values: List[float] = []
    current_wins: List[float] = []
    best_wins: List[float] = []
    vector_missing = Counter()

    for sample in samples:
        rejection_counts.update(
            str(reason)
            for reason in (sample.get("difficulty") or {}).get("rejection_reasons") or []
        )
        failure_mode_counts.update(
            str(mode)
            for mode in list(sample.get("primary_failure_modes") or [])
            + list(sample.get("secondary_failure_modes") or [])
        )
        for key in FAILURE_KEYS:
            value = _nullable_float((sample.get("failure_vector") or {}).get(key))
            if value is not None:
                failure_values[key].append(value)
        changed_factor_counts.update(
            str(factor) for factor in sample.get("changed_factors") or []
        )
        seed_counts[str(sample.get("seed"))] += 1
        run_counts[str(sample.get("source_run"))] += 1
        vector = dict(sample.get("candidate_scenario_vector") or {})
        for key in SCENARIO_VECTOR_KEYS:
            if _nullable_float(vector.get(key)) is None:
                vector_missing[key] += 1
        vector_signatures[_vector_signature(vector)].append(sample)
        value = _nullable_float((sample.get("difficulty") or {}).get("final_value_score"))
        if value is not None:
            values.append(value)
        current = _nullable_float((sample.get("policy_eval") or {}).get("W_current"))
        best = _nullable_float((sample.get("policy_eval") or {}).get("W_best"))
        if current is not None:
            current_wins.append(current)
        if best is not None:
            best_wins.append(best)

    duplicate_groups = [group for group in vector_signatures.values() if len(group) > 1]
    leakage_groups = []
    for group in duplicate_groups:
        splits = sorted(set(str(item.get("split")) for item in group))
        if len(splits) > 1:
            leakage_groups.append(
                {
                    "signature": _vector_signature(
                        group[0].get("candidate_scenario_vector") or {}
                    ),
                    "count": len(group),
                    "splits": splits,
                }
            )
    total = max(len(samples), 1)
    failure_feature_stats = {
        key: _distribution(failure_values.get(key, [])) for key in FAILURE_KEYS
    }
    degenerate_failure_features = [
        key
        for key, stats in failure_feature_stats.items()
        if stats.get("std") is not None and float(stats["std"]) < 1e-8
    ]
    return {
        "schema_version": "falcon.fsn_dataset_audit.v1",
        "total_samples": len(samples),
        "label_counts": _ordered_counts(label_counts, LABELS),
        "accepted_count": label_counts.get("accepted", 0),
        "rejected_count": sum(
            count for label, count in label_counts.items() if label.startswith("rejected_")
        ),
        "invalid_count": label_counts.get("invalid", 0),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "failure_mode_counts": dict(sorted(failure_mode_counts.items())),
        "failure_feature_stats": failure_feature_stats,
        "degenerate_failure_features": degenerate_failure_features,
        "changed_factor_counts": dict(sorted(changed_factor_counts.items())),
        "scenario_vector_missing_rate": {
            key: round(vector_missing.get(key, 0) / total, 6)
            for key in SCENARIO_VECTOR_KEYS
        },
        "value_score": _distribution(values),
        "W_current": _distribution(current_wins),
        "W_best": _distribution(best_wins),
        "seed_counts": dict(sorted(seed_counts.items())),
        "source_run_counts": dict(sorted(run_counts.items())),
        "split_counts": dict(
            sorted(Counter(str(item.get("split")) for item in samples).items())
        ),
        "duplicate_scenario_vector_groups": len(duplicate_groups),
        "duplicate_scenario_vector_samples": sum(len(group) for group in duplicate_groups),
        "cross_split_leakage_group_count": len(leakage_groups),
        "cross_split_leakage_detected": bool(leakage_groups),
        "cross_split_leakage_examples": leakage_groups[:20],
        "warnings": _audit_warnings(
            label_counts, leakage_groups, degenerate_failure_features
        ),
    }


def write_feature_stats(
    samples: Sequence[Mapping[str, Any]], output_path: str | Path
) -> None:
    rows: List[Dict[str, Any]] = []
    feature_paths = [
        *(("failure_vector", key) for key in FAILURE_KEYS),
        *(("proxy_features", key) for key in PROXY_FEATURE_KEYS),
        *(("candidate_scenario_vector", key) for key in SCENARIO_VECTOR_KEYS),
        *(("policy_eval", key) for key in POLICY_KEYS),
        ("difficulty", "final_value_score"),
        ("difficulty", "learning_potential"),
        ("difficulty", "diversity_score"),
    ]
    for section, key in feature_paths:
        values = [
            _nullable_float((item.get(section) or {}).get(key)) for item in samples
        ]
        clean = [value for value in values if value is not None]
        distribution = _distribution(clean)
        rows.append(
            {
                "feature": f"{section}.{key}",
                "count": len(clean),
                "missing_count": len(values) - len(clean),
                "missing_rate": round((len(values) - len(clean)) / max(len(values), 1), 6),
                **distribution,
            }
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "feature",
                "count",
                "missing_count",
                "missing_rate",
                "mean",
                "std",
                "min",
                "max",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def _policy_evals(
    policy: Mapping[str, Any], difficulty: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    current = dict(policy.get("current_policy_eval") or {})
    best = dict(policy.get("best_policy_eval") or {})
    metadata = dict(difficulty.get("metadata") or {})
    if not current:
        current = dict(metadata.get("current_policy_eval") or {})
    if not best:
        best = dict(metadata.get("best_policy_eval") or {})
    return current, best


def _label(accepted: bool, constraint_valid: bool, reasons: Sequence[Any]) -> str:
    if not constraint_valid:
        return "invalid"
    if accepted:
        return "accepted"
    normalized = [str(reason).lower() for reason in reasons]
    if any("too_easy" in reason for reason in normalized):
        return "rejected_too_easy"
    if any("not_solvable" in reason for reason in normalized):
        return "rejected_not_solvable"
    if any("divers" in reason for reason in normalized):
        return "rejected_low_diversity"
    return "rejected_not_solvable"


def _sample_weight(label: str, value_score: Optional[float]) -> float:
    if label == "accepted":
        return round(1.0 + max(value_score or 0.0, 0.0), 6)
    if label == "invalid":
        return 0.1
    return 0.3


def _generator_type(value: Any) -> str:
    text = str(value or "unknown").lower()
    if "qwen" in text or "ollama" in text or "llm" in text:
        return "qwen"
    if "random" in text:
        return "random"
    if "replay" in text or "failure" in text:
        return "replay"
    if "fsn" in text:
        return "fsn"
    return "unknown"


def _seed_from_path(path: Path) -> Optional[int]:
    match = re.search(r"seed[_-](\d+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _seed_split(seed: Optional[int]) -> str:
    if seed in (0, 1, 2) or seed is None:
        return "train"
    if seed == 3:
        return "val"
    if seed == 4:
        return "test"
    return "train"


def _hash_split(sample_id: str) -> str:
    bucket = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def _sample_id(
    source_run: str,
    seed: Optional[int],
    round_id: Optional[int],
    scenario_id: str,
) -> str:
    text = f"{source_run}|{seed}|{round_id}|{scenario_id}"
    return "fsn_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _vector_signature(vector: Mapping[str, Any]) -> str:
    values = []
    for key in SCENARIO_VECTOR_KEYS:
        value = _nullable_float(vector.get(key))
        values.append(None if value is None else round(value, 5))
    return hashlib.sha256(json.dumps(values).encode("utf-8")).hexdigest()[:20]


def _distribution(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(values), 6),
        "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _audit_warnings(
    label_counts: Mapping[str, int],
    leakage_groups: Sequence[Mapping[str, Any]],
    degenerate_failure_features: Sequence[str],
) -> List[str]:
    warnings: List[str] = []
    if label_counts.get("invalid", 0) < 20:
        warnings.append(
            "Invalid class is underrepresented; constraint validity must remain an external check."
        )
    if leakage_groups:
        warnings.append(
            "Exact scenario_vector duplicates cross train/val/test splits. "
            "This smoke split is suitable for interface validation, not a final generalization claim."
        )
    if degenerate_failure_features:
        warnings.append(
            "Failure features with zero variance were found: "
            + ", ".join(degenerate_failure_features)
            + "."
        )
    return warnings


def _ordered_counts(counter: Mapping[str, int], keys: Sequence[str]) -> Dict[str, int]:
    return {key: int(counter.get(key, 0)) for key in keys}


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, MappingABC) else {}


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _invalid_factor(reason: str) -> str:
    mapping = {
        "altitude_below_minimum": "own_center_z",
        "altitude_above_maximum": "own_center_z",
        "velocity_above_maximum": "velocity_difference",
        "team_center_distance_below_minimum": "team_center_distance",
        "team_center_distance_above_maximum": "team_center_distance",
        "own_formation_spread_below_minimum": "own_formation_spread",
        "opponent_formation_spread_above_maximum": "opponent_formation_spread",
        "heading_difference_out_of_range": "heading_difference",
        "aspect_angle_out_of_range": "approximate_aspect_angle",
        "minimum_safety_separation_violation": "own_formation_spread",
    }
    return mapping.get(reason, "team_center_distance")


def _nullable_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
