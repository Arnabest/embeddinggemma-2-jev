#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for Gemma-Jev inference primitives."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.engine import GemmaPrepEngine


def test_heuristic_fallback():
    engine = GemmaPrepEngine(
        base_model_dir="models/base",
        adapter_path="models/adapter/adapter.pt"
    )

    # 1. Noul test
    noul_res = engine.jev_decide(
        mode="noul",
        question="Detecting critical OutOfMemory crash hazard",
        options=[],
        state="VRAM usage exceeds 95%"
    )
    assert "pred" in noul_res
    assert noul_res["pred"] == 1

    # 2. Choice test
    choice_res = engine.jev_decide(
        mode="choice",
        question="Select dispatch route:",
        options=[
            {"t": "Fallback to CPU", "d": "Prevent OOM"},
            {"t": "Force CUDA allocation", "d": "Crash system"}
        ],
        state="GPU VRAM saturated"
    )
    assert "pred" in choice_res
    assert choice_res["pred"] == 0

    # 3. Intent test
    intent_res = engine.classify_intent("Please refactor train.py to support multi-gpu")
    assert intent_res["intent"] == "CODE_MUTATE"

    print("[Tests] All inference primitive unit tests passed successfully!")


if __name__ == "__main__":
    test_heuristic_fallback()
