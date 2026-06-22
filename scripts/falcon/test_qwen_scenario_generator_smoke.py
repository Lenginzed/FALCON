#!/usr/bin/env python
"""Compatibility wrapper for the current Ollama qwen3:8b smoke test.

FALCON currently uses Ollama + qwen3:8b for local LLM scenario generation.
Qwen3-14B and Qwen3-32B are intentionally not used by this script.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from scripts.falcon.test_ollama_qwen8b_scenario_generator_smoke import main  # noqa: E402


if __name__ == "__main__":
    main()
