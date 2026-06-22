"""Post-run analysis utilities for FALCON pilot outputs."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

POSTRUN_REPORT_SCHEMA_VERSION = "falcon.postrun_report.v1"


class FalconPostRunAnalyzer:
    """Analyze a completed FALCON short pilot without running new training."""

    def __init__(self, output_dir: Union[str, Path]) -> None:
        self.output_dir = Path(output_dir)
        self.warnings: List[str] = []
        self.files_read: List[str] = []
        self._summary: Optional[Dict[str, Any]] = None
        self._rounds: Optional[List[Dict[str, Any]]] = None
        self._pool: Optional[Dict[str, Any]] = None
        self._registry: Optional[Dict[str, Any]] = None
        self._round_metrics: Optional[List[Dict[str, Any]]] = None

    def load_summary(self) -> Dict[str, Any]:
        if self._summary is None:
            self._summary = self._load_json("falcon_short_pilot_summary.json")
        return dict(self._summary)

    def load_rounds_csv(self) -> List[Dict[str, Any]]:
        if self._rounds is not None:
            return [dict(item) for item in self._rounds]
        path = self.output_dir / "falcon_short_pilot_rounds.csv"
        if not path.exists():
            self.warnings.append(f"Missing rounds dashboard CSV: {path}")
            self._rounds = []
            return []
        rows: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    rows.append({key: _parse_csv_value(value) for key, value in row.items()})
            self._record_file(path)
        except Exception as exc:  # noqa: BLE001
            self.warnings.append(f"Failed to read rounds CSV {path}: {type(exc).__name__}: {exc}")
        self._rounds = rows
        return [dict(item) for item in rows]

    def load_curriculum_pool(self) -> Dict[str, Any]:
        if self._pool is None:
            self._pool = self._load_json("falcon_curriculum_pool_final.json")
        return dict(self._pool)

    def load_checkpoint_registry(self) -> Dict[str, Any]:
        if self._registry is None:
            self._registry = self._load_json("falcon_checkpoint_registry.json")
        return dict(self._registry)

    def analyze_stability(self) -> Dict[str, Any]:
        summary = self.load_summary()
        stable = bool(summary.get("all_rounds_finished") is True and summary.get("failure_stage") is None)
        return {
            "schema_version": "falcon.postrun_stability.v1",
            "stable": stable,
            "completed_rounds": _int(summary.get("completed_rounds")),
            "all_rounds_finished": bool(summary.get("all_rounds_finished")),
            "failure_stage": summary.get("failure_stage"),
            "total_runtime_seconds": _float(summary.get("total_runtime_seconds")),
            "qwen_failure_count": _int(summary.get("qwen_failure_count")),
            "policy_eval_failure_count": _int(summary.get("policy_eval_failure_count")),
            "difficulty_filter_empty_count": _int(summary.get("difficulty_filter_empty_count")),
        }

    def analyze_curriculum_pool(self) -> Dict[str, Any]:
        summary = self.load_summary()
        pool = self.load_curriculum_pool()
        items = _list_of_mappings(pool.get("items"))
        stats = _mapping(pool.get("stats"))
        accepted = [item for item in items if item.get("accepted_into_curriculum_pool")]
        invalid_count = self._invalid_candidate_count()
        source_counts = Counter(str(item.get("source") or "unknown") for item in items)
        target_modes: Counter[str] = Counter()
        priorities = Counter(str(item.get("priority_level") or "low") for item in items)
        values = [_float(item.get("final_value_score")) for item in items]
        for item in items:
            target_modes.update(str(mode) for mode in item.get("target_failure_modes") or [])
        final_pool_size = len(items) if items else _int(summary.get("final_pool_size"))
        accepted_pool_size = len(accepted) if items else _int(summary.get("accepted_pool_size"))
        accepted_rate = accepted_pool_size / max(final_pool_size, 1)
        round_metrics = self._build_round_metrics()
        pool_sizes = [
            {"round_id": item.get("round_id"), "pool_size": item.get("pool_size")}
            for item in round_metrics
        ]
        accepted_sizes = [
            {"round_id": item.get("round_id"), "accepted_pool_size": item.get("accepted_pool_size")}
            for item in round_metrics
        ]
        pool_size_values = [_int(item.get("pool_size")) for item in round_metrics]
        return {
            "schema_version": "falcon.postrun_curriculum_pool.v1",
            "final_pool_size": final_pool_size,
            "accepted_pool_size": accepted_pool_size,
            "accepted_rate": round(accepted_rate, 6),
            "rejected_count": max(final_pool_size - accepted_pool_size, 0),
            "invalid_count": invalid_count,
            "source_counts": dict(sorted((stats.get("source_counts") or source_counts).items())),
            "target_failure_mode_counts": dict(
                sorted((stats.get("target_failure_mode_counts") or target_modes).items())
            ),
            "mean_value_score": round(
                _float(stats.get("mean_value_score"))
                if stats.get("mean_value_score") is not None
                else (sum(values) / len(values) if values else 0.0),
                6,
            ),
            "high_priority_count": _int(priorities.get("high")),
            "medium_priority_count": _int(priorities.get("medium")),
            "low_priority_count": _int(priorities.get("low")),
            "pool_size_per_round": pool_sizes,
            "accepted_pool_size_per_round": accepted_sizes,
            "pool_growth": (
                pool_size_values[-1] - pool_size_values[0]
                if len(pool_size_values) >= 2
                else (pool_size_values[0] if pool_size_values else 0)
            ),
            "pool_growth_monotonic": all(
                later >= earlier for earlier, later in zip(pool_size_values, pool_size_values[1:])
            ),
        }

    def analyze_fallbacks(self) -> Dict[str, Any]:
        summary = self.load_summary()
        round_metrics = self._build_round_metrics()
        completed_rounds = max(_int(summary.get("completed_rounds")), len(round_metrics))
        fallback_failure_count = _int(summary.get("fallback_failure_used_count"))
        training_fallback_count = _int(summary.get("training_fallback_used_count"))
        fallback_rounds = sum(
            1
            for item in round_metrics
            if item.get("fallback_failure_used") or item.get("training_fallback_used")
        )
        fallback_reasons: Counter[str] = Counter()
        for item in round_metrics:
            for reason in item.get("fallback_reasons") or []:
                fallback_reasons[str(reason)] += 1
        return {
            "schema_version": "falcon.postrun_fallbacks.v1",
            "fallback_failure_used_count": fallback_failure_count,
            "training_fallback_used_count": training_fallback_count,
            "fallback_round_count": fallback_rounds,
            "fallback_rate": round(fallback_rounds / max(completed_rounds, 1), 6),
            "per_round": [
                {
                    "round_id": item.get("round_id"),
                    "fallback_failure_used": item.get("fallback_failure_used"),
                    "training_fallback_used": item.get("training_fallback_used"),
                    "fallback_reasons": item.get("fallback_reasons") or [],
                }
                for item in round_metrics
            ],
            "fallback_reason_counts": dict(sorted(fallback_reasons.items())),
        }

    def analyze_qwen_generation(self) -> Dict[str, Any]:
        summary = self.load_summary()
        round_metrics = self._build_round_metrics()
        generated = _int(summary.get("total_candidates_generated"))
        validated = _int(summary.get("total_candidates_validated"))
        accepted = _int(summary.get("total_candidates_accepted"))
        qwen_generated = sum(_int(item.get("qwen_candidates")) for item in round_metrics)
        qwen_accepted = sum(_int(item.get("qwen_candidates_accepted")) for item in round_metrics)
        return {
            "schema_version": "falcon.postrun_qwen_generation.v1",
            "total_candidates_generated": generated,
            "total_candidates_validated": validated,
            "total_candidates_accepted": accepted,
            "qwen_candidates_generated": qwen_generated,
            "qwen_candidates_accepted": qwen_accepted,
            "qwen_candidate_acceptance_rate": round(qwen_accepted / max(qwen_generated, 1), 6),
            "per_round_qwen_accepted": [
                {
                    "round_id": item.get("round_id"),
                    "qwen_candidates": item.get("qwen_candidates"),
                    "qwen_candidates_accepted": item.get("qwen_candidates_accepted"),
                    "qwen_failed": item.get("qwen_failed"),
                }
                for item in round_metrics
            ],
        }

    def analyze_policy_eval(self) -> Dict[str, Any]:
        round_metrics = self._build_round_metrics()
        learning_values: List[float] = []
        better_count = 0
        for item in round_metrics:
            learning_values.extend(item.get("learning_potentials") or [])
            better_count += _int(item.get("w_best_gt_current_count"))
        return {
            "schema_version": "falcon.postrun_policy_eval.v1",
            "per_round": [
                {
                    "round_id": item.get("round_id"),
                    "current_win_rate": item.get("current_win_rate"),
                    "best_win_rate": item.get("best_win_rate"),
                    "mean_learning_potential": item.get("mean_learning_potential"),
                    "max_learning_potential": item.get("max_learning_potential"),
                    "w_best_gt_current_count": item.get("w_best_gt_current_count"),
                    "num_policy_eval_success": item.get("num_policy_eval_success"),
                    "policy_eval_failures": item.get("policy_eval_failures"),
                }
                for item in round_metrics
            ],
            "mean_learning_potential": round(sum(learning_values) / len(learning_values), 6) if learning_values else 0.0,
            "max_learning_potential": round(max(learning_values), 6) if learning_values else 0.0,
            "w_best_gt_current_candidate_count": better_count,
            "has_w_best_gt_current_candidates": better_count > 0,
        }

    def analyze_training_progress(self) -> Dict[str, Any]:
        summary = self.load_summary()
        registry = self.load_checkpoint_registry()
        round_metrics = self._build_round_metrics()
        resume_path = summary.get("resume_state_path")
        resume_supported = bool(resume_path and (self.output_dir / Path(str(resume_path)).name).exists())
        checkpoints_by_round = []
        for item in round_metrics:
            checkpoints_by_round.append(
                {
                    "round_id": item.get("round_id"),
                    "checkpoint_saved": item.get("checkpoint_saved"),
                    "checkpoint_path": item.get("checkpoint_path"),
                }
            )
        return {
            "schema_version": "falcon.postrun_training_progress.v1",
            "total_checkpoints_saved": _int(summary.get("total_checkpoints_saved")),
            "latest_checkpoint_path": summary.get("latest_checkpoint_path") or registry.get("latest_checkpoint"),
            "best_checkpoint_path": summary.get("best_checkpoint_path") or registry.get("best_checkpoint"),
            "per_round_checkpoints": checkpoints_by_round,
            "resume_supported": resume_supported,
            "resume_state_path": resume_path,
        }

    def export_report(self, output_path: Union[str, Path]) -> Dict[str, Any]:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stability = self.analyze_stability()
        curriculum = self.analyze_curriculum_pool()
        fallbacks = self.analyze_fallbacks()
        qwen = self.analyze_qwen_generation()
        policy_eval = self.analyze_policy_eval()
        training = self.analyze_training_progress()
        diagnostics = self._diagnose(stability, curriculum, fallbacks, qwen, policy_eval, training)
        report = {
            "schema_version": POSTRUN_REPORT_SCHEMA_VERSION,
            "output_dir": str(self.output_dir),
            "analyzed_at": _timestamp(),
            "files_read": sorted(set(self.files_read)),
            "stability": stability,
            "curriculum_pool": curriculum,
            "fallbacks": fallbacks,
            "qwen_generation": qwen,
            "policy_eval": policy_eval,
            "training_progress": training,
            "diagnostics": diagnostics,
            "warnings": sorted(set(self.warnings)),
        }
        _write_json(output_path, report)
        txt_path = output_path.with_name("falcon_postrun_report.txt")
        txt_path.write_text(self._render_text_report(report), encoding="utf-8-sig")
        csv_path = output_path.with_name("falcon_postrun_round_metrics.csv")
        self._write_round_metrics_csv(csv_path, self._build_round_metrics())
        return report

    def _build_round_metrics(self) -> List[Dict[str, Any]]:
        if self._round_metrics is not None:
            return [dict(item) for item in self._round_metrics]
        dashboard_by_round = {
            _int(item.get("round_id")): item for item in self.load_rounds_csv()
        }
        round_ids = sorted(
            {
                *dashboard_by_round.keys(),
                *[
                    _round_id_from_name(path.name)
                    for path in self.output_dir.glob("falcon_controller_difficulty_round*.json")
                    if _round_id_from_name(path.name) is not None
                ],
            }
        )
        metrics: List[Dict[str, Any]] = []
        for round_id in round_ids:
            dashboard = dashboard_by_round.get(round_id, {})
            difficulty_data = self._load_json(f"falcon_controller_difficulty_round{round_id}.json", required=False)
            difficulty = _list_of_mappings(difficulty_data.get("difficulty_results"))
            generation = self._load_json(f"falcon_controller_candidates_round{round_id}.json", required=False)
            candidates = _list_of_mappings(generation.get("candidates"))
            training = self._load_json(f"falcon_controller_training_round{round_id}_summary.json", required=False)
            failure = self._load_json(f"falcon_controller_failure_summary_round{round_id}.json", required=False)
            learning = [_float(item.get("learning_potential")) for item in difficulty]
            better_count = sum(
                1
                for item in difficulty
                if _float(_mapping(_mapping(item.get("metadata")).get("best_policy_eval")).get("win_rate"))
                > _float(_mapping(_mapping(item.get("metadata")).get("current_policy_eval")).get("win_rate"))
            )
            qwen_accepted = sum(
                1
                for idx, item in enumerate(difficulty)
                if item.get("accepted_into_curriculum_pool")
                and idx < len(candidates)
                and _is_qwen_candidate(candidates[idx])
            )
            fallback_reasons = []
            if training.get("fallback_reason"):
                fallback_reasons.append(str(training.get("fallback_reason")))
            failure_source = _mapping(failure.get("failure_source"))
            failure_type = str(failure_source.get("type") or "")
            if failure_type and failure_type != "real_failure_trajectory":
                fallback_reasons.append(f"failure_source={failure_type}")
            train_summary = _mapping(training.get("train_summary"))
            checkpoint_path = train_summary.get("actor_checkpoint_path") or dashboard.get("current_checkpoint")
            metrics.append(
                {
                    "round_id": round_id,
                    "train_steps": _int(dashboard.get("train_steps")),
                    "eval_episodes": _int(dashboard.get("eval_episodes")),
                    "qwen_candidates": (
                        sum(1 for candidate in candidates if _is_qwen_candidate(candidate))
                        if candidates
                        else _int(dashboard.get("qwen_candidates"))
                    ),
                    "num_schema_valid": _int(dashboard.get("num_schema_valid")),
                    "num_constraint_valid": _int(dashboard.get("num_constraint_valid")),
                    "num_policy_eval_success": _int(dashboard.get("num_policy_eval_success")),
                    "num_accepted": _int(dashboard.get("num_accepted")),
                    "qwen_candidates_accepted": qwen_accepted,
                    "fallback_failure_used": bool(_int(dashboard.get("fallback_failure_used"))),
                    "training_fallback_used": bool(_int(dashboard.get("training_fallback_used"))),
                    "fallback_reasons": fallback_reasons,
                    "current_checkpoint": dashboard.get("current_checkpoint"),
                    "best_checkpoint": dashboard.get("best_checkpoint"),
                    "current_win_rate": _float(dashboard.get("round_win_rate")),
                    "best_win_rate": _float(dashboard.get("best_win_rate")),
                    "learning_potentials": learning,
                    "mean_learning_potential": round(sum(learning) / len(learning), 6) if learning else 0.0,
                    "max_learning_potential": round(max(learning), 6) if learning else 0.0,
                    "w_best_gt_current_count": better_count,
                    "pool_size": _int(dashboard.get("pool_size")),
                    "accepted_pool_size": _int(dashboard.get("accepted_pool_size")),
                    "qwen_failed": bool(_int(dashboard.get("qwen_failed"))),
                    "policy_eval_failures": _int(dashboard.get("policy_eval_failures")),
                    "difficulty_filter_empty": bool(_int(dashboard.get("difficulty_filter_empty"))),
                    "checkpoint_saved": bool(
                        train_summary.get("checkpoint_saved")
                        or (round_id == 0 and dashboard.get("current_checkpoint"))
                    ),
                    "checkpoint_path": checkpoint_path,
                }
            )
        self._round_metrics = metrics
        return [dict(item) for item in metrics]

    def _invalid_candidate_count(self) -> int:
        invalid_ids = set()
        for path in sorted(self.output_dir.glob("falcon_controller_validated_candidates_round*.json")):
            data = self._load_json(path.name, required=False)
            for item in _list_of_mappings(data.get("schema_validations")):
                if not item.get("is_valid"):
                    invalid_ids.add(str(item.get("scenario_id") or f"{path.name}:schema"))
            for item in _list_of_mappings(data.get("constraint_results")):
                if not item.get("is_valid"):
                    invalid_ids.add(str(item.get("scenario_id") or f"{path.name}:constraint"))
        return len(invalid_ids)

    def _diagnose(
        self,
        stability: Mapping[str, Any],
        curriculum: Mapping[str, Any],
        fallbacks: Mapping[str, Any],
        qwen: Mapping[str, Any],
        policy_eval: Mapping[str, Any],
        training: Mapping[str, Any],
    ) -> Dict[str, Any]:
        risks: List[str] = []
        recommendations: List[str] = []
        accepted_rate = _float(curriculum.get("accepted_rate"))
        completed_rounds = _int(stability.get("completed_rounds"))
        training_fallback_count = _int(fallbacks.get("training_fallback_used_count"))
        if accepted_rate < 0.1:
            risks.append("accepted 场景比例偏低，可能需要增加 Qwen candidates、放宽 hard filter 或训练更强 best checkpoint。")
        if training_fallback_count > completed_rounds * 0.5:
            risks.append("训练 fallback 偏多，sampling plan 中 accepted 场景不足。")
        if _int(stability.get("qwen_failure_count")) > 0:
            risks.append("Qwen 生成存在失败，需要检查 Ollama 或 JSON 输出。")
        if _int(stability.get("policy_eval_failure_count")) > 0:
            risks.append("策略评估存在失败，需要检查 env load 或 checkpoint rollout。")
        if _int(stability.get("difficulty_filter_empty_count")) > completed_rounds * 0.5:
            risks.append("超过一半 round 的 difficulty hard filter 为空，课程供给仍不稳定。")
        if not policy_eval.get("has_w_best_gt_current_candidates"):
            risks.append("未观察到 W_best > W_current 的候选场景，双边界筛选信号可能不足。")
        stable = bool(stability.get("stable"))
        recommend_medium = bool(
            stable
            and _int(stability.get("qwen_failure_count")) == 0
            and _int(stability.get("policy_eval_failure_count")) == 0
            and _float(fallbacks.get("fallback_rate")) <= 0.5
            and accepted_rate >= 0.1
            and bool(training.get("resume_supported"))
        )
        if recommend_medium:
            recommendations.append(
                "建议进入 medium pilot：保持单 seed，将 round 数提高到 8-10，并将每轮训练步数提高到 512-1024。"
            )
            recommendations.append(
                "保持每轮 2 个 Qwen 候选；若连续两个 round hard filter 为空，再提高到 3-4 个候选。"
            )
            recommendations.append(
                "将每候选 policy eval episode 提高到 3，以降低 best checkpoint 更新和 hard filter 的方差。"
            )
        else:
            recommendations.append("暂不建议进入 medium pilot，先处理主要风险并再运行一次 short pilot。")
        return {
            "schema_version": "falcon.postrun_diagnostics.v1",
            "stable": stable,
            "recommend_medium_pilot": recommend_medium,
            "main_risks": risks,
            "recommendations": recommendations,
        }

    def _render_text_report(self, report: Mapping[str, Any]) -> str:
        stability = _mapping(report.get("stability"))
        curriculum = _mapping(report.get("curriculum_pool"))
        fallbacks = _mapping(report.get("fallbacks"))
        qwen = _mapping(report.get("qwen_generation"))
        policy = _mapping(report.get("policy_eval"))
        training = _mapping(report.get("training_progress"))
        diagnostics = _mapping(report.get("diagnostics"))
        lines = [
            "FALCON Short Pilot Post-Run Report",
            "=" * 40,
            f"Output directory: {report.get('output_dir')}",
            f"Stable: {diagnostics.get('stable')}",
            f"Recommend medium pilot: {diagnostics.get('recommend_medium_pilot')}",
            "",
            "Stability",
            f"- Completed rounds: {stability.get('completed_rounds')}",
            f"- All rounds finished: {stability.get('all_rounds_finished')}",
            f"- Failure stage: {stability.get('failure_stage')}",
            f"- Runtime seconds: {stability.get('total_runtime_seconds')}",
            f"- Qwen failures: {stability.get('qwen_failure_count')}",
            f"- Policy eval failures: {stability.get('policy_eval_failure_count')}",
            f"- Empty difficulty rounds: {stability.get('difficulty_filter_empty_count')}",
            "",
            "Curriculum Pool",
            f"- Final pool size: {curriculum.get('final_pool_size')}",
            f"- Accepted pool size: {curriculum.get('accepted_pool_size')}",
            f"- Accepted rate: {curriculum.get('accepted_rate')}",
            f"- Rejected / invalid: {curriculum.get('rejected_count')} / {curriculum.get('invalid_count')}",
            f"- Mean value score: {curriculum.get('mean_value_score')}",
            f"- Pool growth / monotonic: {curriculum.get('pool_growth')} / {curriculum.get('pool_growth_monotonic')}",
            "",
            "Fallbacks",
            f"- Failure fallback count: {fallbacks.get('fallback_failure_used_count')}",
            f"- Training fallback count: {fallbacks.get('training_fallback_used_count')}",
            f"- Fallback rate: {fallbacks.get('fallback_rate')}",
            "",
            "Qwen / Policy Evaluation",
            f"- Generated / validated / accepted: {qwen.get('total_candidates_generated')} / {qwen.get('total_candidates_validated')} / {qwen.get('total_candidates_accepted')}",
            f"- Qwen candidate acceptance rate: {qwen.get('qwen_candidate_acceptance_rate')}",
            f"- Mean / max learning potential: {policy.get('mean_learning_potential')} / {policy.get('max_learning_potential')}",
            f"- Has W_best > W_current candidates: {policy.get('has_w_best_gt_current_candidates')}",
            "",
            "Training Progress",
            f"- Total checkpoints saved: {training.get('total_checkpoints_saved')}",
            f"- Latest checkpoint: {training.get('latest_checkpoint_path')}",
            f"- Best checkpoint: {training.get('best_checkpoint_path')}",
            f"- Resume supported: {training.get('resume_supported')}",
            "",
            "Main Risks",
        ]
        risks = diagnostics.get("main_risks") or ["No major risk detected by current rules."]
        lines.extend(f"- {item}" for item in risks)
        lines.extend(["", "Recommendations"])
        lines.extend(f"- {item}" for item in diagnostics.get("recommendations") or [])
        return "\n".join(lines) + "\n"

    def _write_round_metrics_csv(self, path: Path, metrics: Sequence[Mapping[str, Any]]) -> None:
        fieldnames = [
            "round_id",
            "train_steps",
            "eval_episodes",
            "qwen_candidates",
            "num_schema_valid",
            "num_constraint_valid",
            "num_policy_eval_success",
            "num_accepted",
            "qwen_candidates_accepted",
            "fallback_failure_used",
            "training_fallback_used",
            "fallback_reasons",
            "current_win_rate",
            "best_win_rate",
            "mean_learning_potential",
            "max_learning_potential",
            "w_best_gt_current_count",
            "pool_size",
            "accepted_pool_size",
            "qwen_failed",
            "policy_eval_failures",
            "difficulty_filter_empty",
            "checkpoint_saved",
            "checkpoint_path",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in metrics:
                row = {key: item.get(key) for key in fieldnames}
                row["fallback_reasons"] = " | ".join(item.get("fallback_reasons") or [])
                writer.writerow(row)

    def _load_json(self, filename: str, required: bool = True) -> Dict[str, Any]:
        path = self.output_dir / filename
        if not path.exists():
            if required:
                self.warnings.append(f"Missing expected post-run input: {path}")
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self._record_file(path)
            return dict(data) if isinstance(data, MappingABC) else {}
        except Exception as exc:  # noqa: BLE001
            self.warnings.append(f"Failed to read JSON {path}: {type(exc).__name__}: {exc}")
            return {}

    def _record_file(self, path: Path) -> None:
        value = str(path)
        if value not in self.files_read:
            self.files_read.append(value)


def _is_qwen_candidate(candidate: Mapping[str, Any]) -> bool:
    value = str(candidate.get("generator_type") or "").lower()
    return "qwen" in value or "ollama" in value


def _round_id_from_name(name: str) -> Optional[int]:
    marker = "round"
    if marker not in name:
        return None
    suffix = name.rsplit(marker, 1)[-1].split(".", 1)[0]
    try:
        return int(suffix)
    except ValueError:
        return None


def _parse_csv_value(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, MappingABC) else {}


def _list_of_mappings(value: Any) -> List[Mapping[str, Any]]:
    return [item for item in (value or []) if isinstance(item, MappingABC)]


def _float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def _int(value: Any) -> int:
    return int(round(_float(value)))


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, sort_keys=True)
