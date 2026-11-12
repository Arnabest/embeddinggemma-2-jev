#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JEV Decision Model Adapter (Gemma-2-JEV 因果微内核适配器).

Binds to Gemma-2-JEV PyTorch adapter and HTTP daemon, with deterministic heuristic fallback.
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
from deciders.directional_gate import evaluate_directional_tool_gate, is_readonly_tool_call


class JevGemmaDecisionAdapter(BaseDecisionModelAdapter):
    """JEV Causal Decider Adapter connecting to Gemma-2-JEV engine / daemon."""

    def __init__(self, fallback: Optional[BaseDecisionModelAdapter] = None):
        self.fallback = fallback or HeuristicDecisionAdapter()
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from engine import GemmaPrepEngine
                self._engine = GemmaPrepEngine()
            except Exception:
                self._engine = None
        return self._engine

    @property
    def model_name(self) -> str:
        return "jev_gemma_causal_engine"

    def infer_intent(self, prompt: str, context: str = "") -> IntentDecision:
        eng = self._get_engine()
        if eng:
            try:
                res = eng.classify_intent(prompt, context)
                intent_name = res.get("intent", "CODE_MUTATE")
                rec_skills = ["pre-action-audit"] if intent_name == "DIRECTIONAL_PLANNING" else []
                return IntentDecision(
                    intent=intent_name,
                    confidence=res.get("confidence", 0.85),
                    recommended_skills=rec_skills,
                    description=res.get("description", ""),
                    slots=res.get("slots", {}),
                    raw=res,
                    requires_clarification=bool(res.get("requires_clarification") or intent_name == "DIRECTIONAL_PLANNING"),
                )
            except Exception:
                pass
        return self.fallback.infer_intent(prompt, context)

    def evaluate_hazard(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> HazardDecision:
        if context and context.get("directional_locked"):
            gate = evaluate_directional_tool_gate(tool_name, args, locked=True)
            if gate:
                return HazardDecision(
                    decision="deny",
                    risk_level="HIGH",
                    reason=gate["reason"],
                    is_blocked=True,
                )
        if is_readonly_tool_call(tool_name, args):
            return HazardDecision(decision="allow", risk_level="LOW")

        eng = self._get_engine()
        if eng:
            try:
                # 1. Hardware VRAM saturation check
                hw = context if (context and "danger_level" in context) else eng.get_hardware_status()
                if hw and hw.get("danger_level", 1) >= 5 and tool_name == "run_command":
                    cmd = str(args.get("CommandLine", "")).lower()
                    if any(k in cmd for k in ("python", "torch", "comfy", "train")):
                        return HazardDecision(
                            decision="ask",
                            risk_level="HIGH",
                            reason=f"Gemma-Jev hardware guard: VRAM usage is at {hw.get('vram_usage_pct')}%. Launching large compute jobs may cause CUDA OOM.",
                            requires_human_confirmation=True,
                            details=hw,
                        )

                # 2. Critical file overwrite check
                if tool_name in ("write_file", "write_to_file"):
                    target = str(args.get("Path") or args.get("TargetFile") or "").lower()
                    is_overwrite = bool(args.get("Overwrite"))
                    if is_overwrite and any(k in target for k in ("config", "adapter.pt", "model.safetensors")):
                        dec = eng.jev_decide(
                            "noul",
                            question="检测到系统关键模型或配置文件覆盖写入，是否判定为潜在高危行为？",
                            options=[],
                            state=f"Target: {target}, Overwrite: {is_overwrite}",
                        )
                        if dec.get("pred") == 1:
                            return HazardDecision(
                                decision="ask",
                                risk_level="HIGH",
                                reason=f"Gemma-Jev safety guard: Overwriting critical asset [{target}] requires confirmation.",
                                requires_human_confirmation=True,
                                details=dec,
                            )

                # 3. Command safety audit
                if tool_name == "run_command":
                    cmd = args.get("CommandLine", "")
                    cwd = args.get("Cwd", "")
                    daemon = args.get("IsDaemon", False)
                    res = eng.audit_command_safety(cmd, cwd, daemon)
                    level = res.get("danger_level", 1)
                    if level >= 5:
                        return HazardDecision(
                            decision="deny",
                            risk_level="CRITICAL",
                            reason=res.get("reason", "Dangerous command intercepted by JEV guard"),
                            is_blocked=True,
                            details=res,
                        )
                    elif level >= 3:
                        return HazardDecision(
                            decision="ask",
                            risk_level="HIGH",
                            reason=res.get("reason", "Medium-to-high impact command requires human confirmation"),
                            requires_human_confirmation=True,
                            details=res,
                        )
                    return HazardDecision(decision="allow", risk_level="LOW", details=res)
            except Exception:
                pass
        return self.fallback.evaluate_hazard(tool_name, args, context)

    def triage_failure(
        self,
        traceback_text: str,
        exit_code: int = 1,
    ) -> TriageDecision:
        eng = self._get_engine()
        if eng:
            try:
                res = eng.triage_failure(traceback_text, exit_code)
                return TriageDecision(
                    primary_cause=res.get("primary_cause", "Unknown Failure"),
                    component=res.get("component", "system"),
                    confidence=res.get("confidence", 0.8),
                    suggested_fix=res.get("suggested_fix", ""),
                    raw=res,
                )
            except Exception:
                pass
        return self.fallback.triage_failure(traceback_text, exit_code)

    def arbitrate_divergence(
        self,
        branches: List[Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> DivergenceDecision:
        eng = self._get_engine()
        if eng and branches:
            try:
                opts = [{"t": str(b), "d": f"Branch candidate {i}"} for i, b in enumerate(branches)]
                dec = eng.jev_decide(
                    mode="choice",
                    question="Select optimal architectural branch for execution",
                    options=opts,
                    state=str(context or ""),
                )
                pred_idx = dec.get("pred", 0)
                if 0 <= pred_idx < len(branches):
                    chosen = str(branches[pred_idx])
                    return DivergenceDecision(
                        chosen_branch=chosen,
                        rationale=f"JEV choice predicted branch index {pred_idx}",
                        confidence=dec.get("conf", 0.85),
                        discarded_branches=[str(b) for i, b in enumerate(branches) if i != pred_idx],
                    )
            except Exception:
                pass
        return self.fallback.arbitrate_divergence(branches, context)
