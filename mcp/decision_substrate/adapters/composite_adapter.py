#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Composite Decision Model Adapter (多层级级联降级适配器).

Chains primary decision model -> secondary decision model -> deterministic heuristic fallback.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..protocol import (
    BaseDecisionModelAdapter,
    DivergenceDecision,
    HazardDecision,
    IntentDecision,
    TriageDecision,
)
from .heuristic_adapter import HeuristicDecisionAdapter


class CompositeDecisionAdapter(BaseDecisionModelAdapter):
    """Multi-tiered composite adapter that cascades across primary and fallback engines."""

    def __init__(self, primary: BaseDecisionModelAdapter, fallback: Optional[BaseDecisionModelAdapter] = None):
        self.primary = primary
        self.fallback = fallback or HeuristicDecisionAdapter()

    @property
    def model_name(self) -> str:
        return f"composite({self.primary.model_name} -> {self.fallback.model_name})"

    def infer_intent(self, prompt: str, context: str = "") -> IntentDecision:
        try:
            return self.primary.infer_intent(prompt, context)
        except Exception:
            return self.fallback.infer_intent(prompt, context)

    def evaluate_hazard(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> HazardDecision:
        try:
            return self.primary.evaluate_hazard(tool_name, args, context)
        except Exception:
            return self.fallback.evaluate_hazard(tool_name, args, context)

    def triage_failure(
        self,
        traceback_text: str,
        exit_code: int = 1,
    ) -> TriageDecision:
        try:
            return self.primary.triage_failure(traceback_text, exit_code)
        except Exception:
            return self.fallback.triage_failure(traceback_text, exit_code)

    def arbitrate_divergence(
        self,
        branches: List[Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> DivergenceDecision:
        try:
            return self.primary.arbitrate_divergence(branches, context)
        except Exception:
            return self.fallback.arbitrate_divergence(branches, context)
