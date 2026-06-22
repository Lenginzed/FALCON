"""Ollama/Qwen offline scenario generator for FALCON.

This module asks a local LLM service to propose CandidateScenario objects for
the offline curriculum loop. It never participates in real-time decisions.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Mapping as MappingABC
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .candidate_schema import CANDIDATE_SCHEMA_VERSION, create_candidate_scenario, validate_candidate_schema
from .constraint_checker import ConstraintChecker
from .scenario_adapter import (
    extract_initial_config_from_yaml,
    initial_config_to_scenario_vector,
    load_base_scenario_config,
    scenario_vector_to_initial_config,
)
from .trajectory_recorder import SCENARIO_VECTOR_KEYS

QWEN_GENERATOR_SCHEMA_VERSION = "falcon.qwen_scenario_generator.v1"
QWEN_GENERATION_RESULT_SCHEMA_VERSION = "falcon.qwen_generation_result.v1"

DEFAULT_ALLOWED_PARAMETER_SPACE: Dict[str, Any] = {
    "team_center_distance": [6000.0, 18000.0],
    "own_formation_spread": [1000.0, 8000.0],
    "opponent_formation_spread": [1000.0, 8000.0],
    "altitude_difference": [-3000.0, 3000.0],
    "velocity_difference": [-80.0, 80.0],
    "heading_difference": [0.0, 2.0 * math.pi],
    "approximate_aspect_angle": [0.0, 2.0 * math.pi],
    "altitude": [3000.0, 9000.0],
    "velocity": [180.0, 300.0],
    "minimum_separation": 500.0,
    "max_perturbation_level": 0.35,
    "max_changed_factors": 3,
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "provider": "ollama",
    "provider_mode": "ollama_native",
    "base_url_native": "http://localhost:11434",
    "base_url_openai": "http://localhost:11434/v1",
    "base_url": None,
    "model_name": "qwen3:8b",
    "model": "qwen3:8b",
    "temperature": 0.1,
    "top_p": 0.8,
    "max_tokens": 4096,
    "timeout": 180.0,
    "stream": False,
    "think": False,
    "reasoning_effort": "none",
    "num_retries": 2,
    "api_key": None,
    "generator_type": "ollama_qwen3_8b_teacher",
    "prompt_template_path": "falcon/prompts/qwen_scenario_generation_prompt.txt",
    "allowed_parameter_space": DEFAULT_ALLOWED_PARAMETER_SPACE,
    "constraint_checker": {},
}


class QwenScenarioGenerator:
    """Generate FALCON CandidateScenario objects with local Qwen via Ollama.

    The LLM is used only in an offline curriculum-generation loop. The generated
    objects are repaired locally when possible, then validated and constrained
    before downstream evaluation.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = _deep_merge(DEFAULT_CONFIG, dict(config or {}))
        self._normalize_model_config()
        model_text = f"{self.config.get('model_name', '')} {self.config.get('model', '')} {self.config.get('local_model_path', '')}"
        model_text_lower = model_text.lower()
        if "32b" in model_text_lower:
            raise ValueError("FALCON currently forbids Qwen3-32B in this interface; use Ollama qwen3:8b.")
        if "14b" in model_text_lower:
            raise ValueError("FALCON currently uses Ollama qwen3:8b; Qwen3-14B is not enabled in this run.")
        self.allowed_parameter_space = _deep_merge(
            DEFAULT_ALLOWED_PARAMETER_SPACE,
            dict(self.config.get("allowed_parameter_space") or {}),
        )
        self.config["allowed_parameter_space"] = self.allowed_parameter_space
        self.last_result: Dict[str, Any] = {
            "schema_version": QWEN_GENERATION_RESULT_SCHEMA_VERSION,
            "candidates": [],
            "warnings": [],
        }

    def check_llm_server(self) -> Dict[str, Any]:
        """Check local LLM service health without touching vLLM by default."""
        provider = str(self.config.get("provider", "ollama")).lower()
        if provider != "ollama":
            return {
                "schema_version": "falcon.llm_health_check.v1",
                "provider": provider,
                "provider_mode": self.config.get("provider_mode"),
                "base_url": self._active_base_url(),
                "model_name": self.config.get("model_name"),
                "server_reachable": None,
                "model_available": None,
                "warnings": ["Only Ollama health checks are implemented for the current FALCON smoke path."],
            }
        url = f"{str(self.config.get('base_url_native')).rstrip('/')}/api/tags"
        warnings: List[str] = []
        models: List[str] = []
        server_reachable = False
        model_available = False
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
            with urllib.request.urlopen(request, timeout=float(self.config.get("timeout", 180.0))) as response:
                data = json.loads(response.read().decode("utf-8"))
            server_reachable = True
            models = _ollama_model_names(data)
            model_available = str(self.config.get("model_name")) in models
            if not model_available:
                warnings.append(f"Model {self.config.get('model_name')} is not available. Run: ollama pull qwen3:8b")
        except urllib.error.URLError as exc:
            warnings.append(f"Ollama service is not reachable at {url}: {exc.reason}")
        except json.JSONDecodeError as exc:
            warnings.append(f"Ollama /api/tags returned non-JSON data: {exc}")
        return {
            "schema_version": "falcon.llm_health_check.v1",
            "provider": provider,
            "provider_mode": self.config.get("provider_mode"),
            "base_url": self.config.get("base_url_native"),
            "model_name": self.config.get("model_name"),
            "server_reachable": server_reachable,
            "model_available": model_available,
            "models": models,
            "warnings": warnings,
        }

    def generate_from_failure_summary(
        self,
        failure_summary: Mapping[str, Any],
        base_config: Mapping[str, Any],
        num_scenarios: int = 5,
        pool_stats: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Ask the configured offline LLM for candidates and return validated candidates only."""
        warnings: List[str] = []
        raw_responses: List[Dict[str, Any]] = []
        attempts: List[Dict[str, Any]] = []
        candidates: List[Dict[str, Any]] = []
        retry_feedback: Optional[str] = None
        max_attempts = max(1, int(self.config.get("num_retries", 2)) + 1)

        for attempt in range(max_attempts):
            messages = self._build_messages(
                failure_summary=failure_summary,
                base_config=base_config,
                num_scenarios=num_scenarios,
                pool_stats=pool_stats,
                retry_feedback=retry_feedback,
            )
            try:
                llm_response = self._call_llm(messages)
            except RuntimeError as exc:
                message = str(exc)
                warnings.append(message)
                raw_responses.append({"attempt": attempt, "error": message, "content": ""})
                break

            raw_text = str(llm_response.get("content", ""))
            if llm_response.get("thinking_detected"):
                warnings.append("LLM thinking output was detected and ignored during parsing.")
            raw_responses.append(
                {
                    "attempt": attempt,
                    "error": None,
                    "content": raw_text,
                    "raw_response": llm_response.get("raw_response"),
                    "thinking_detected": bool(llm_response.get("thinking_detected")),
                    "provider": self.config.get("provider"),
                    "provider_mode": self.config.get("provider_mode"),
                }
            )
            repaired = self.repair_or_retry_invalid_response(
                raw_text=llm_response.get("raw_response") or raw_text,
                failure_summary=failure_summary,
                base_config=base_config,
                num_scenarios=num_scenarios,
            )
            validation = self.validate_and_filter_candidates(
                repaired.get("candidates", []),
                base_config=base_config,
                failure_summary=failure_summary,
            )
            attempt_record = {
                "schema_version": "falcon.qwen_generation_attempt.v1",
                "attempt": attempt,
                "parse_result": repaired.get("parse_result"),
                "repaired_candidate_count": len(repaired.get("candidates", [])),
                "schema_validations": validation.get("schema_validations", []),
                "constraint_results": validation.get("constraint_results", []),
                "valid_candidate_count": len(validation.get("valid_candidates", [])),
                "warnings": sorted(set(repaired.get("warnings", []) + validation.get("warnings", []))),
            }
            attempts.append(attempt_record)
            candidates = validation.get("valid_candidates", [])[:num_scenarios]
            if len(candidates) >= int(num_scenarios):
                warnings.extend(attempt_record["warnings"])
                break
            if candidates:
                warnings.append(
                    f"Only {len(candidates)} valid candidates were produced; retrying for requested {int(num_scenarios)}."
                )
            retry_feedback = _validation_feedback(attempt_record)

        if not candidates and not warnings:
            warnings.append(f"{self.config.get('model_name')} returned no usable candidates after repair and retry.")

        self.last_result = {
            "schema_version": QWEN_GENERATION_RESULT_SCHEMA_VERSION,
            "provider": self.config.get("provider"),
            "provider_mode": self.config.get("provider_mode"),
            "model_name": self.config.get("model_name"),
            "model": self.config.get("model_name"),
            "base_url": self._active_base_url(),
            "generator_type": self.config.get("generator_type"),
            "requested_num_scenarios": int(num_scenarios),
            "candidates": candidates,
            "raw_responses": raw_responses,
            "attempts": attempts,
            "thinking_detected": any(item.get("thinking_detected") for item in raw_responses),
            "warnings": sorted(set(warnings)),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        return candidates

    def generate_from_failure_file(
        self,
        failure_summary_path: Union[str, Path],
        base_config_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """Load failure summary and base YAML, then generate candidates."""
        with Path(failure_summary_path).open("r", encoding="utf-8") as f:
            failure_summary = json.load(f)
        base_config = load_base_scenario_config(base_config_path)
        candidates = self.generate_from_failure_summary(failure_summary, base_config)
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.last_result, f, indent=2, sort_keys=True)
        return candidates

    def parse_llm_response(self, raw_text: Any) -> Dict[str, Any]:
        """Parse an LLM response, tolerating Ollama native wrappers and prose."""
        warnings: List[str] = []
        parsed: Any = None
        content_text, thinking_detected, provider_warnings = _content_from_provider_response(raw_text)
        warnings.extend(provider_warnings)
        stripped_content, stripped_thinking = _strip_thinking(content_text)
        thinking_detected = thinking_detected or stripped_thinking
        if stripped_thinking:
            warnings.append("Removed <think>...</think> content before JSON parsing.")
        json_text = stripped_content.strip()
        if not json_text:
            return {
                "schema_version": "falcon.qwen_parse_result.v1",
                "is_valid_json": False,
                "candidates": [],
                "json_text": "",
                "thinking_detected": thinking_detected,
                "warnings": sorted(set(warnings + ["Qwen response was empty."])),
            }

        for candidate_text in _json_text_candidates(json_text):
            try:
                parsed = json.loads(candidate_text)
                json_text = candidate_text
                break
            except json.JSONDecodeError:
                continue

        if parsed is None:
            return {
                "schema_version": "falcon.qwen_parse_result.v1",
                "is_valid_json": False,
                "candidates": [],
                "json_text": "",
                "thinking_detected": thinking_detected,
                "warnings": sorted(set(warnings + ["Could not parse a valid JSON object or array from Qwen response."])),
            }

        candidates = _payload_to_candidate_list(parsed)
        if not candidates:
            warnings.append("Parsed JSON did not contain a candidates list or CandidateScenario object.")
        if json_text != stripped_content.strip():
            warnings.append("Extracted JSON from surrounding non-JSON text.")
        return {
            "schema_version": "falcon.qwen_parse_result.v1",
            "is_valid_json": True,
            "candidates": candidates,
            "json_text": json_text,
            "thinking_detected": thinking_detected,
            "warnings": warnings,
        }

    def repair_or_retry_invalid_response(
        self,
        raw_text: str,
        failure_summary: Optional[Mapping[str, Any]] = None,
        base_config: Optional[Mapping[str, Any]] = None,
        num_scenarios: int = 5,
        validation_errors: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Locally repair parseable Qwen candidates; generation handles API retry."""
        parse_result = self.parse_llm_response(raw_text)
        warnings = list(parse_result.get("warnings") or [])
        if validation_errors:
            warnings.append("Repair received validation feedback: " + "; ".join(str(item) for item in validation_errors))
        if not parse_result.get("is_valid_json"):
            return {
                "schema_version": "falcon.qwen_repair_result.v1",
                "candidates": [],
                "parse_result": parse_result,
                "warnings": warnings,
            }
        if base_config is None:
            warnings.append("base_config was missing; local CandidateScenario repair cannot build initial_config.")
            base_initial_config = {"agents": []}
            base_vector = {}
        else:
            base_initial_config = extract_initial_config_from_yaml(base_config)
            base_vector = initial_config_to_scenario_vector(base_initial_config)["scenario_vector"]
        repaired_candidates: List[Dict[str, Any]] = []
        for idx, candidate in enumerate(parse_result.get("candidates", [])[: max(0, int(num_scenarios))]):
            repaired, candidate_warnings = self._repair_candidate(
                candidate,
                idx=idx,
                failure_summary=failure_summary or {},
                base_initial_config=base_initial_config,
                base_vector=base_vector,
            )
            repaired_candidates.append(repaired)
            warnings.extend(candidate_warnings)
        return {
            "schema_version": "falcon.qwen_repair_result.v1",
            "candidates": repaired_candidates,
            "parse_result": parse_result,
            "warnings": sorted(set(warnings)),
        }

    def validate_and_filter_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        base_config: Mapping[str, Any],
        failure_summary: Optional[Mapping[str, Any]] = None,
        constraint_checker: Optional[ConstraintChecker] = None,
    ) -> Dict[str, Any]:
        """Validate CandidateScenario schema and physical/task constraints."""
        warnings: List[str] = []
        checker = constraint_checker or ConstraintChecker(self.config.get("constraint_checker"))
        base_initial_config = extract_initial_config_from_yaml(base_config)
        base_vector = initial_config_to_scenario_vector(base_initial_config)["scenario_vector"]
        normalized_candidates: List[Dict[str, Any]] = []
        schema_validations: List[Dict[str, Any]] = []
        constraint_results: List[Dict[str, Any]] = []
        valid_candidates: List[Dict[str, Any]] = []

        for idx, candidate in enumerate(candidates):
            if not isinstance(candidate, MappingABC):
                candidate = {}
            repaired = dict(candidate)
            if not isinstance(repaired.get("initial_config"), MappingABC) or not repaired.get("initial_config", {}).get("agents"):
                repaired, repair_warnings = self._repair_candidate(
                    repaired,
                    idx=idx,
                    failure_summary=failure_summary or {},
                    base_initial_config=base_initial_config,
                    base_vector=base_vector,
                )
                warnings.extend(repair_warnings)
            validation = validate_candidate_schema(repaired)
            schema_validations.append({"scenario_id": repaired.get("scenario_id"), **validation})
            if not validation.get("is_valid"):
                warnings.append(f"Candidate {repaired.get('scenario_id', idx)} failed schema validation: {validation.get('missing_fields')}")
                normalized_candidates.append(repaired)
                constraint_results.append(
                    {
                        "schema_version": "1.0",
                        "scenario_id": str(repaired.get("scenario_id", idx)),
                        "is_valid": False,
                        "validity_score": 0.0,
                        "rejection_reasons": ["candidate_schema_invalid"],
                        "physical_constraint_check": {},
                        "task_constraint_check": {},
                        "missing_fields": validation.get("missing_fields", []),
                        "warnings": validation.get("warnings", []),
                    }
                )
                continue
            constraint = checker.validate_candidate(repaired)
            constraint_results.append(constraint)
            normalized_candidates.append(repaired)
            if constraint.get("is_valid"):
                valid_candidates.append(repaired)
            else:
                warnings.append(f"Candidate {repaired.get('scenario_id')} failed constraint checks: {constraint.get('rejection_reasons')}")

        return {
            "schema_version": "falcon.qwen_candidate_filter_result.v1",
            "candidates": normalized_candidates,
            "valid_candidates": valid_candidates,
            "schema_validations": schema_validations,
            "constraint_results": constraint_results,
            "warnings": sorted(set(warnings)),
        }

    def _build_messages(
        self,
        failure_summary: Mapping[str, Any],
        base_config: Mapping[str, Any],
        num_scenarios: int,
        pool_stats: Optional[Mapping[str, Any]] = None,
        retry_feedback: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        prompt = self._render_prompt(
            failure_summary=failure_summary,
            base_config=base_config,
            num_scenarios=num_scenarios,
            pool_stats=pool_stats,
        )
        if retry_feedback:
            prompt += (
                "\n\nYour previous response failed validation. "
                "Return repaired JSON only. Validation feedback:\n"
                + retry_feedback
            )
        return [
            {
                "role": "system",
                "content": (
                    "You are an offline FALCON training scenario generator. "
                    "You are not a controller. Do not output thinking. Return JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def _normalize_model_config(self) -> None:
        model_name = self.config.get("model_name") or self.config.get("model") or "qwen3:8b"
        self.config["model_name"] = str(model_name)
        self.config["model"] = str(model_name)
        if self.config.get("base_url") and not self.config.get("base_url_openai"):
            self.config["base_url_openai"] = self.config["base_url"]
        provider = str(self.config.get("provider", "ollama")).lower()
        provider_mode = str(self.config.get("provider_mode", "ollama_native")).lower()
        self.config["provider"] = provider
        self.config["provider_mode"] = provider_mode

    def _active_base_url(self) -> str:
        provider_mode = str(self.config.get("provider_mode", "ollama_native")).lower()
        if provider_mode == "ollama_native":
            return str(self.config.get("base_url_native", "http://localhost:11434"))
        return str(self.config.get("base_url") or self.config.get("base_url_openai", "http://localhost:11434/v1"))

    def _render_prompt(
        self,
        failure_summary: Mapping[str, Any],
        base_config: Mapping[str, Any],
        num_scenarios: int,
        pool_stats: Optional[Mapping[str, Any]] = None,
    ) -> str:
        template_path = Path(str(self.config.get("prompt_template_path")))
        if not template_path.is_absolute():
            template_path = Path.cwd() / template_path
        if not template_path.exists():
            template_path = Path(__file__).resolve().parent / "prompts" / "qwen_scenario_generation_prompt.txt"
        template = Template(template_path.read_text(encoding="utf-8"))
        base_initial_config = extract_initial_config_from_yaml(base_config)
        base_vector = initial_config_to_scenario_vector(base_initial_config)["scenario_vector"]
        compact_failure = _compact_failure_summary(failure_summary)
        if pool_stats:
            compact_failure["pool_stats"] = _jsonable(pool_stats)
        return template.safe_substitute(
            num_scenarios=int(num_scenarios),
            candidate_schema_json=json.dumps(_candidate_schema_for_prompt(), indent=2, sort_keys=True),
            allowed_parameter_space_json=json.dumps(self.allowed_parameter_space, indent=2, sort_keys=True),
            base_scenario_vector_json=json.dumps(base_vector, indent=2, sort_keys=True),
            failure_summary_json=json.dumps(compact_failure, indent=2, sort_keys=True),
        )

    def _call_llm(self, messages: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
        provider = str(self.config.get("provider", "ollama")).lower()
        provider_mode = str(self.config.get("provider_mode", "ollama_native")).lower()
        if provider == "ollama" and provider_mode == "ollama_native":
            return self._call_ollama_native(messages)
        return self._call_openai_compatible(messages)

    def _call_ollama_native(self, messages: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
        base_url = str(self.config.get("base_url_native", "http://localhost:11434")).rstrip("/")
        url = f"{base_url}/api/chat"
        payload = {
            "model": self.config.get("model_name", "qwen3:8b"),
            "messages": list(messages),
            "think": bool(self.config.get("think", False)),
            "stream": bool(self.config.get("stream", False)),
            "options": {
                "temperature": float(self.config.get("temperature", 0.1)),
                "top_p": float(self.config.get("top_p", 0.8)),
                "num_predict": int(self.config.get("max_tokens", 4096)),
            },
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=float(self.config.get("timeout", 180.0))) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Ollama native API HTTP error at {url}: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Ollama native API is not reachable at "
                f"{url}. Start Ollama and ensure the model exists with: ollama pull qwen3:8b. "
                f"Original error: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(f"Ollama native API request timed out at {url}.") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama native API returned non-JSON response at {url}.") from exc
        content, thinking_detected, warnings = _content_from_provider_response(response_data)
        if warnings and not content:
            raise RuntimeError("; ".join(warnings))
        return {
            "content": content,
            "raw_response": response_data,
            "thinking_detected": thinking_detected,
        }

    def _call_openai_compatible(self, messages: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
        base_url = str(self.config.get("base_url") or self.config.get("base_url_openai", "http://localhost:11434/v1")).rstrip("/")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": self.config.get("model_name", "qwen3:8b"),
            "messages": list(messages),
            "temperature": float(self.config.get("temperature", 0.1)),
            "top_p": float(self.config.get("top_p", 0.8)),
            "max_tokens": int(self.config.get("max_tokens", 4096)),
            "stream": bool(self.config.get("stream", False)),
            "reasoning_effort": self.config.get("reasoning_effort", "none"),
            "reasoning": {"effort": self.config.get("reasoning_effort", "none")},
        }
        body = json.dumps(payload).encode("utf-8")
        api_key = self.config.get("api_key") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=float(self.config.get("timeout", 180.0))) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Ollama OpenAI-compatible API HTTP error at {url}: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Ollama OpenAI-compatible API is not reachable at "
                f"{url}. Start Ollama and ensure the model exists with: ollama pull qwen3:8b. "
                f"Original error: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(f"Ollama OpenAI-compatible API request timed out at {url}.") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama OpenAI-compatible API returned non-JSON response at {url}.") from exc

        choices = response_data.get("choices") if isinstance(response_data, MappingABC) else None
        if not choices:
            raise RuntimeError("Ollama OpenAI-compatible API response did not contain choices.")
        message = choices[0].get("message") if isinstance(choices[0], MappingABC) else None
        content = message.get("content") if isinstance(message, MappingABC) else choices[0].get("text")
        if not isinstance(content, str):
            raise RuntimeError("Ollama OpenAI-compatible API response did not contain text content.")
        thinking_detected = bool(isinstance(message, MappingABC) and message.get("thinking")) or bool(re.search(r"<think\b", content, flags=re.IGNORECASE))
        return {
            "content": content,
            "raw_response": response_data,
            "thinking_detected": thinking_detected,
        }

    def _repair_candidate(
        self,
        candidate: Mapping[str, Any],
        idx: int,
        failure_summary: Mapping[str, Any],
        base_initial_config: Mapping[str, Any],
        base_vector: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        warnings: List[str] = []
        raw = dict(candidate or {})
        raw_vector = raw.get("scenario_vector")
        if not isinstance(raw_vector, MappingABC):
            params = raw.get("scenario_parameters")
            raw_vector = params.get("scenario_vector") if isinstance(params, MappingABC) else {}
        vector = {}
        for key in SCENARIO_VECTOR_KEYS:
            source_value = raw_vector.get(key) if isinstance(raw_vector, MappingABC) else None
            if source_value is None:
                source_value = base_vector.get(key)
                warnings.append(f"Candidate {idx} missing scenario_vector.{key}; filled from base scenario.")
            vector[key] = self._clip_scenario_value(key, source_value, warnings, idx)

        source_failure_id = raw.get("source_failure_id") or _source_failure_id(failure_summary)
        changed_factors = _list_or_empty(raw.get("changed_factors"))
        changed_factors = [key for key in changed_factors if key in SCENARIO_VECTOR_KEYS][: int(self.allowed_parameter_space.get("max_changed_factors", 3))]
        if not changed_factors:
            changed_factors = _changed_factors_from_vectors(vector, base_vector)
            if not changed_factors:
                changed_factors = ["team_center_distance"]
            warnings.append(f"Candidate {idx} changed_factors was missing or invalid; inferred {changed_factors}.")

        target_modes = _list_or_empty(raw.get("target_failure_modes"))
        if not target_modes:
            target_modes = _list_or_empty(failure_summary.get("primary_failure_modes")) or _list_or_empty(failure_summary.get("secondary_failure_modes"))
            warnings.append(f"Candidate {idx} target_failure_modes was missing; filled from failure_summary.")

        scenario_id = str(raw.get("scenario_id") or f"ollama_qwen3_8b_{idx:04d}")
        metadata = dict(raw.get("metadata") or {})
        metadata.update(
            {
                "provider": self.config.get("provider"),
                "provider_mode": self.config.get("provider_mode"),
                "llm_model": self.config.get("model_name"),
                "offline_generation_only": True,
                "qwen14b_used": False,
                "qwen32b_used": False,
                "think_disabled_requested": not bool(self.config.get("think", False)),
                "repaired_by_falcon": True,
            }
        )
        scenario_parameters = dict(raw.get("scenario_parameters") or {})
        scenario_parameters.setdefault("allowed_parameter_space", self.allowed_parameter_space)
        try:
            initial_config = scenario_vector_to_initial_config(vector, base_initial_config)
        except Exception as exc:  # noqa: BLE001 - repair must not crash generation
            initial_config = None
            warnings.append(f"Candidate {idx} initial_config generation failed: {exc}")

        repaired = create_candidate_scenario(
            scenario_id=scenario_id,
            generator_type=str(self.config.get("generator_type", "qwen3_14b_teacher")),
            source_failure_id=str(source_failure_id) if source_failure_id is not None else None,
            target_failure_modes=target_modes,
            changed_factors=changed_factors,
            counterfactual_group_id=raw.get("counterfactual_group_id") or f"cf_{_safe_id(source_failure_id)}",
            scenario_vector=vector,
            scenario_parameters=scenario_parameters,
            initial_config=initial_config,
            expected_effect=raw.get("expected_effect"),
            rationale=raw.get("rationale"),
            metadata=metadata,
        )
        return repaired, warnings

    def _clip_scenario_value(self, key: str, value: Any, warnings: List[str], idx: int) -> Optional[float]:
        numeric = _float_or_none(value)
        if numeric is None:
            return None
        bounds = self.allowed_parameter_space.get(key)
        if isinstance(bounds, Sequence) and not isinstance(bounds, (str, bytes)) and len(bounds) >= 2:
            low = _float_or_none(bounds[0])
            high = _float_or_none(bounds[1])
            if low is not None and high is not None:
                clipped = max(low, min(high, numeric))
                if clipped != numeric:
                    warnings.append(f"Candidate {idx} scenario_vector.{key} clipped from {numeric} to {clipped}.")
                return clipped
        return numeric


def _candidate_schema_for_prompt() -> Dict[str, Any]:
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "required_fields": [
            "schema_version",
            "scenario_id",
            "generator_type",
            "source_failure_id",
            "target_failure_modes",
            "changed_factors",
            "counterfactual_group_id",
            "scenario_vector",
            "scenario_parameters",
            "initial_config",
            "expected_effect",
            "rationale",
            "metadata",
        ],
        "scenario_vector_keys": list(SCENARIO_VECTOR_KEYS),
    }


def _compact_failure_summary(failure_summary: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "schema_version",
        "source_trajectory",
        "failure_scores",
        "primary_failure_modes",
        "secondary_failure_modes",
        "severity_level",
        "failure_severity",
        "submetrics",
        "evidence",
        "scenario_vector",
        "episode_summary",
    )
    compact = {key: _jsonable(failure_summary.get(key)) for key in keys if key in failure_summary}
    if "failure_severity" not in compact and isinstance(failure_summary.get("failure_scores"), MappingABC):
        compact["failure_severity"] = failure_summary["failure_scores"].get("failure_severity")
    return compact


def _json_text_candidates(text: str) -> List[str]:
    cleaned = text.strip()
    candidates = [cleaned]
    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(block.strip() for block in fenced)
    for start_char in ("{", "["):
        start = cleaned.find(start_char)
        while start >= 0:
            end = _matching_json_end(cleaned, start)
            if end >= 0:
                candidates.append(cleaned[start : end + 1])
                break
            start = cleaned.find(start_char, start + 1)
    unique = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _matching_json_end(text: str, start: int) -> int:
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    stack = [closer]
    in_string = False
    escape = False
    for idx in range(start + 1, len(text)):
        char = text[idx]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return -1
            stack.pop()
            if not stack:
                return idx
    return -1


def _payload_to_candidate_list(payload: Any) -> List[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, MappingABC)]
    if isinstance(payload, MappingABC):
        candidates = payload.get("candidates")
        if isinstance(candidates, list):
            return [item for item in candidates if isinstance(item, MappingABC)]
        if "scenario_id" in payload or "scenario_vector" in payload:
            return [payload]
    return []


def _content_from_provider_response(raw_response: Any) -> Tuple[str, bool, List[str]]:
    warnings: List[str] = []
    thinking_detected = False
    if isinstance(raw_response, MappingABC):
        if "candidates" in raw_response or "scenario_vector" in raw_response or "scenario_id" in raw_response:
            return json.dumps(raw_response), thinking_detected, warnings
        message = raw_response.get("message")
        if isinstance(message, MappingABC):
            thinking_detected = message.get("thinking") is not None
            content = message.get("content")
            if isinstance(content, str):
                return content, thinking_detected, warnings
        if raw_response.get("thinking") is not None:
            thinking_detected = True
        content = raw_response.get("content") or raw_response.get("response")
        if isinstance(content, str):
            return content, thinking_detected, warnings
        choices = raw_response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, MappingABC):
                choice_message = choice.get("message")
                if isinstance(choice_message, MappingABC):
                    thinking_detected = thinking_detected or choice_message.get("thinking") is not None
                    content = choice_message.get("content")
                    if isinstance(content, str):
                        return content, thinking_detected, warnings
                content = choice.get("text")
                if isinstance(content, str):
                    return content, thinking_detected, warnings
        warnings.append("Provider response did not include message.content or text content.")
        return "", thinking_detected, warnings
    if isinstance(raw_response, str):
        stripped = raw_response.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, MappingABC) and any(key in parsed for key in ("message", "choices", "content", "response", "thinking")):
                    content, parsed_thinking, parsed_warnings = _content_from_provider_response(parsed)
                    if content:
                        return content, thinking_detected or parsed_thinking, parsed_warnings
                if isinstance(parsed, MappingABC) and any(key in parsed for key in ("candidates", "scenario_vector", "scenario_id")):
                    return raw_response, thinking_detected, warnings
                if isinstance(parsed, list):
                    return raw_response, thinking_detected, warnings
            except json.JSONDecodeError:
                pass
        return raw_response, bool(re.search(r"<think\b", raw_response, flags=re.IGNORECASE)), warnings
    return "", thinking_detected, ["LLM response was neither a string nor a mapping."]


def _strip_thinking(content: str) -> Tuple[str, bool]:
    if not isinstance(content, str):
        return "", False
    stripped, count = re.subn(r"<think\b[^>]*>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL)
    return stripped, count > 0


def _ollama_model_names(data: Any) -> List[str]:
    if not isinstance(data, MappingABC):
        return []
    models = data.get("models")
    if not isinstance(models, list):
        return []
    names = []
    for model in models:
        if isinstance(model, MappingABC):
            name = model.get("name") or model.get("model")
            if name is not None:
                names.append(str(name))
    return names


def _source_failure_id(failure_summary: Mapping[str, Any]) -> Optional[str]:
    for key in ("source_failure_id", "source_trajectory", "trajectory_id", "episode_id"):
        value = failure_summary.get(key)
        if value is not None:
            return str(value)
    return None


def _changed_factors_from_vectors(vector: Mapping[str, Any], base_vector: Mapping[str, Any]) -> List[str]:
    changed = []
    for key in SCENARIO_VECTOR_KEYS:
        a = _float_or_none(vector.get(key))
        b = _float_or_none(base_vector.get(key))
        if a is None or b is None:
            continue
        scale = max(abs(b), 1.0)
        if abs(a - b) / scale > 0.05:
            changed.append(key)
    return changed[:3]


def _validation_feedback(attempt_record: Mapping[str, Any]) -> str:
    errors = []
    for validation in attempt_record.get("schema_validations") or []:
        if not validation.get("is_valid"):
            errors.append(f"{validation.get('scenario_id')}: missing {validation.get('missing_fields')}; warnings {validation.get('warnings')}")
    for constraint in attempt_record.get("constraint_results") or []:
        if not constraint.get("is_valid"):
            errors.append(f"{constraint.get('scenario_id')}: rejected {constraint.get('rejection_reasons')}; missing {constraint.get('missing_fields')}")
    if not errors:
        errors.append("No candidates were returned or all candidates were filtered out.")
    return "\n".join(errors[:10])


def _list_or_empty(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str) and value:
        return [value]
    return []


def _safe_id(value: Any) -> str:
    value = str(value or "unknown")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[-80:]


def _jsonable(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, MappingABC) and isinstance(result.get(key), MappingABC):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
