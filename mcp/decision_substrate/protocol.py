#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified Decision Protocol & Data Contracts.

Defines the model-agnostic contract for any decision engine taking over the agent loop:
- Intent classification & skill dispatch
- Tool mutation & runtime hazard guards
- Failure triage & stacktrace attribution
- Divergence & competing execution route arbitration
- Physical ground-truth completion gate verification
- Post-turn codebase line budget & architectural audit
- Dependency routing guidance for code generation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IntentType(str, Enum):
    # 原有核心系统与工程意图
    CODE_MUTATE = "CODE_MUTATE"
    READONLY_QUERY = "READONLY_QUERY"
    SYSTEM_COMMAND = "SYSTEM_COMMAND"
    HIGH_HAZARD = "HIGH_HAZARD"
    GENERAL_CONVERSATION = "GENERAL_CONVERSATION"

    # 扩展补全的系统关键意图 (6大空白)
    BROWSER_ACTION = "BROWSER_ACTION"
    MULTIMODAL_PROCESS = "MULTIMODAL_PROCESS"
    SWARM_ORCHESTRATE = "SWARM_ORCHESTRATE"
    STATE_ROLLBACK = "STATE_ROLLBACK"
    DEEP_RESEARCH = "DEEP_RESEARCH"
    COMPOUND_CHAIN = "COMPOUND_CHAIN"
    DIRECTIONAL_PLANNING = "DIRECTIONAL_PLANNING"


@dataclass
class IntentDecision:
    """Pre-turn intention parsing outcome."""
    intent: str
    confidence: float
    recommended_skills: List[str] = field(default_factory=list)
    risk_level: str = "LOW"
    description: str = ""
    slots: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    requires_clarification: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "recommended_skills": self.recommended_skills,
            "risk_level": self.risk_level,
            "description": self.description,
            "slots": self.slots,
            "raw": self.raw,
            "requires_clarification": self.requires_clarification,
        }


@dataclass
class HazardDecision:
    """In-flight tool execution safety verdict."""
    decision: str  # "allow", "deny", "ask"
    risk_level: str = "LOW"  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    reason: str = ""
    is_blocked: bool = False
    requires_human_confirmation: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_proto_hook_dict(self) -> Dict[str, Any]:
        """Strict schema for Antigravity Go protojson unmarshaler (decision + reason only)."""
        res = {"decision": self.decision}
        if self.reason:
            res["reason"] = self.reason
        return res

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "risk_level": self.risk_level,
            "reason": self.reason,
            "is_blocked": self.is_blocked,
            "requires_human_confirmation": self.requires_human_confirmation,
            "details": self.details,
        }


@dataclass
class TriageDecision:
    """Execution error stacktrace attribution and fix recommendation."""
    primary_cause: str
    component: str
    confidence: float
    suggested_fix: str
    is_fatal: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_cause": self.primary_cause,
            "component": self.component,
            "confidence": self.confidence,
            "suggested_fix": self.suggested_fix,
            "is_fatal": self.is_fatal,
            "raw": self.raw,
        }


@dataclass
class DivergenceDecision:
    """Decision between competing branches or tool calls."""
    chosen_branch: str
    rationale: str
    confidence: float
    discarded_branches: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chosen_branch": self.chosen_branch,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "discarded_branches": self.discarded_branches,
        }


@dataclass
class CompletionDecision:
    """Physical completion gate verification outcome."""
    passed: bool
    reason: str
    risk_type: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "risk_type": self.risk_type,
            "details": self.details,
        }


@dataclass
class PreTurnGrounding:
    """Standardized pre-turn grounding context card."""
    intent: IntentDecision
    epistemic_weights: Dict[str, Any]
    hesitation_status: Dict[str, Any]
    hardware_status: Dict[str, Any]
    recalled_memories: List[Dict[str, Any]]
    hud_markdown: str
    inject_steps: List[Dict[str, Any]] = field(default_factory=list)
    dependency_guidance: List[Dict[str, Any]] = field(default_factory=list)
    directional_gate: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.to_dict(),
            "epistemic_weights": self.epistemic_weights,
            "hesitation_status": self.hesitation_status,
            "hardware_status": self.hardware_status,
            "recalled_memories": self.recalled_memories,
            "hud_markdown": self.hud_markdown,
            "inject_steps": self.inject_steps,
            "dependency_guidance": self.dependency_guidance,
            "directional_gate": self.directional_gate,
        }


@dataclass
class PostTurnAudit:
    """Post-task codebase audit and memory vault payload."""
    status: str  # "PASS", "WARN_REFACTOR", "SYNC_VAULT"
    line_limit: int
    max_lines_exceeded: List[str]
    changes_analyzed: int
    exit_code: int
    vault_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "line_limit": self.line_limit,
            "max_lines_exceeded": self.max_lines_exceeded,
            "changes_analyzed": self.changes_analyzed,
            "exit_code": self.exit_code,
            "vault_payload": self.vault_payload,
        }


class BaseDecisionModelAdapter(ABC):
    """Abstract model adapter defining the decision primitives for any underlying model.
    
    Implementations may be:
    - JEV / Gemma-2 local PyTorch adapter + daemon
    - Local or cloud LLM (DeepSeek, Qwen, Claude, LLaMA) via OpenAI-compatible API
    - Fast heuristic rule engine (zero network, zero weight dependencies)
    - Composite fallback adapter
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the underlying decision model."""
        pass

    @abstractmethod
    def infer_intent(self, prompt: str, context: str = "") -> IntentDecision:
        """Parse user prompt and extract intent, confidence, risk, and recommended skills."""
        pass

    @abstractmethod
    def evaluate_hazard(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> HazardDecision:
        """Evaluate safety risk and return allow / ask / deny."""
        pass

    @abstractmethod
    def triage_failure(
        self,
        traceback_text: str,
        exit_code: int = 1,
    ) -> TriageDecision:
        """Triage exception stacktraces or command failures."""
        pass

    @abstractmethod
    def arbitrate_divergence(
        self,
        branches: List[Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> DivergenceDecision:
        """Select optimal route among competing tool execution paths."""
        pass

    def is_available(self) -> bool:
        """Health probe checking if the underlying model backend is live."""
        return True
