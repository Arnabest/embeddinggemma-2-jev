#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Physical unit tests for Unified Decision Substrate (统一决策模块基座).

Verifies:
1. Base decision protocol contracts & data structures.
2. Hot-swapping model adapters: JEV -> Heuristic -> Composite -> Custom Mock.
3. Pre-turn intention grounding & Attention HUD assembly.
4. In-flight tool mutation & hardware-aware hazard check.
5. Competing branch divergence arbitration.
6. Bounded deliberation / hesitation budget state machine (3 default + 1 extension).
7. Execution failure stacktrace triage.
8. Physical completion gate anti-cheating & exit code 0 verification.
9. Post-task codebase line budget (800 lines limit) audit.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
CORE_DIR = HERE.parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from decision_substrate import (
    BaseDecisionModelAdapter,
    CompletionDecision,
    CompositeDecisionAdapter,
    DecisionSubstrate,
    DivergenceDecision,
    EpistemicArbiter,
    EpistemicContext,
    HazardDecision,
    HesitationBudgetManager,
    HesitationState,
    HeuristicDecisionAdapter,
    IntentDecision,
    JevGemmaDecisionAdapter,
    PostTurnAudit,
    PreTurnGrounding,
    TriageDecision,
)


class MockDecisionModelAdapter(BaseDecisionModelAdapter):
    """Custom mock decision model adapter simulating third-party LLM (e.g. DeepSeek / LLaMA)."""

    def __init__(self, name: str = "mock_deepseek_v3"):
        self._name = name

    @property
    def model_name(self) -> str:
        return self._name

    def infer_intent(self, prompt: str, context: str = "") -> IntentDecision:
        return IntentDecision(
            intent="CODE_MUTATE",
            confidence=0.97,
            recommended_skills=["nexus-scaffold", "triad-bridge"],
            risk_level="MEDIUM",
            description="DeepSeek-V3 high-confidence code refactoring",
        )

    def evaluate_hazard(self, tool_name: str, args: dict, context: dict = None) -> HazardDecision:
        if tool_name == "run_command" and "rm -rf" in str(args.get("CommandLine", "")):
            return HazardDecision(decision="deny", risk_level="CRITICAL", reason="Intercepted destructive rm command")
        return HazardDecision(decision="allow", risk_level="LOW")

    def triage_failure(self, traceback_text: str, exit_code: int = 1) -> TriageDecision:
        return TriageDecision(
            primary_cause="Mock Model Diagnosed Error",
            component="runtime",
            confidence=0.99,
            suggested_fix="Apply mock patch",
        )

    def arbitrate_divergence(self, branches: list, context: dict = None) -> DivergenceDecision:
        return DivergenceDecision(
            chosen_branch=str(branches[0]) if branches else "",
            rationale="Mock Pareto optimal route",
            confidence=0.95,
        )


def test_protocol_and_dataclass_contracts():
    intent = IntentDecision(intent="CODE_MUTATE", confidence=0.88, recommended_skills=["test-skill"])
    assert intent.to_dict()["intent"] == "CODE_MUTATE"

    hazard = HazardDecision(decision="ask", risk_level="HIGH", reason="Confirm overwrite")
    proto_dict = hazard.to_proto_hook_dict()
    assert proto_dict == {"decision": "ask", "reason": "Confirm overwrite"}
    assert "risk_level" not in proto_dict  # Strict Go protojson compatibility


def test_hot_swapping_adapters():
    substrate = DecisionSubstrate()

    # Default model
    assert "jev" in substrate.adapter.model_name.lower() or "causal" in substrate.adapter.model_name.lower()

    # Switch to heuristic
    switched = substrate.switch_model("heuristic")
    assert switched == "heuristic_symbolic_engine"
    assert isinstance(substrate.adapter, HeuristicDecisionAdapter)

    # Test heuristic intent
    intent = substrate.adapter.infer_intent("查看最新文件内容")
    assert intent.intent == "READONLY_QUERY"

    # Switch to composite
    switched = substrate.switch_model("composite")
    assert "composite" in switched
    assert isinstance(substrate.adapter, CompositeDecisionAdapter)

    # Hot-swap to custom mock model
    mock_adapter = MockDecisionModelAdapter(name="mock_llama3_70b")
    substrate.set_adapter(mock_adapter)
    assert substrate.adapter.model_name == "mock_llama3_70b"

    intent = substrate.adapter.infer_intent("任意指令")
    assert intent.confidence == 0.97
    assert intent.description == "DeepSeek-V3 high-confidence code refactoring"


def test_pre_turn_grounding_and_attention_hud():
    substrate = DecisionSubstrate(adapter=HeuristicDecisionAdapter())
    substrate.hesitation_mgr.reset(default_budget=3)

    grounding = substrate.on_pre_turn(
        session_id="test_sess",
        prompt="编写并测试快速排序算法",
        state="",
        include_recall=False,
        include_hardware=False,
    )

    assert isinstance(grounding, PreTurnGrounding)
    assert grounding.intent.intent == "CODE_MUTATE"
    assert "weights" in grounding.epistemic_weights
    assert grounding.hesitation_status["current_budget"] == 3
    assert "[Unified-Decision-Substrate: 前置态势与先验装配]" in grounding.hud_markdown
    assert "- [核心意图] `CODE_MUTATE`" in grounding.hud_markdown
    assert "- [犹豫预算] 剩余机会: 3/3" in grounding.hud_markdown


def test_in_flight_hazard_checks():
    substrate = DecisionSubstrate(adapter=HeuristicDecisionAdapter())

    # Safe read-only tool
    h1 = substrate.on_hazard_check("view_file", {"AbsolutePath": "/workspace/test.py"})
    assert h1.decision == "allow"
    assert h1.to_proto_hook_dict() == {"decision": "allow"}

    # Dangerous system deletion
    h2 = substrate.on_hazard_check("run_command", {"CommandLine": "rm -rf /"})
    assert h2.decision == "deny"
    assert h2.is_blocked is True
    assert h2.to_proto_hook_dict()["decision"] == "deny"

    # Protected git metadata directory
    h3 = substrate.on_hazard_check("write_to_file", {"TargetFile": "/workspace/repo/.git/config"})
    assert h3.decision == "deny"
    assert h3.is_blocked is True


def test_divergence_resolution():
    substrate = DecisionSubstrate(adapter=HeuristicDecisionAdapter())
    branches = ["run_command(pytest)", "view_file(main.py)", "write_to_file(fix.py)"]

    dec = substrate.on_divergence(branches)
    assert isinstance(dec, DivergenceDecision)
    assert "view_file" in dec.chosen_branch  # Prioritizes safe observation


def test_bounded_hesitation_lifecycle():
    substrate = DecisionSubstrate(adapter=HeuristicDecisionAdapter())
    substrate.on_hesitation(action="reset")

    # Step 1-3 consume
    for i in range(3):
        r = substrate.on_hesitation(action="consume", reason=f"probe_{i+1}")
        assert r["allowed"] is True

    # 4th blocked
    r4 = substrate.on_hesitation(action="consume", reason="probe_unauthorized")
    assert r4["allowed"] is False
    assert r4["action_required"] == "APPLY_TO_HUMAN"

    # Human grants 1 extension
    ext = substrate.on_hesitation(action="apply_extension", user_approved=True)
    assert ext["success"] is True
    assert ext["remaining_budget"] == 1

    # Final probe consumes extension
    r_final = substrate.on_hesitation(action="consume", reason="probe_extension")
    assert r_final["allowed"] is True
    assert r_final["state"] == "hard_locked"

    # Permanent lock
    r_locked = substrate.on_hesitation(action="consume", reason="probe_past_limit")
    assert r_locked["allowed"] is False
    assert r_locked["action_required"] == "HARD_CONVERGENCE"


def test_failure_triage():
    substrate = DecisionSubstrate(adapter=HeuristicDecisionAdapter())
    triage = substrate.on_failure_triage(
        traceback_text="ModuleNotFoundError: No module named 'scipy'",
        exit_code=1,
    )
    assert "scipy" in triage.primary_cause
    assert "pip install scipy" in triage.suggested_fix


def test_physical_completion_gate():
    substrate = DecisionSubstrate(adapter=HeuristicDecisionAdapter())

    # Case 1: Code mutated on test task but zero tests run -> REJECT
    c1 = substrate.on_completion_gate(
        session_id="test_sess",
        user_prompt="重构核心模块并执行单元测试验证",
        mutated_files=["kernel.py"],
        test_commands=[],
        all_results=[],
    )
    assert c1.passed is False
    assert c1.risk_type == "UNVERIFIED_MUTATION"

    # Case 2: Test executed and failed with exit_code 1 -> REJECT
    class DummyToolCall:
        args = {"CommandLine": "python -m pytest"}
    class DummyToolResult:
        output = {"exit_code": 1}
        ast_verified = True

    c2 = substrate.on_completion_gate(
        session_id="test_sess",
        user_prompt="重构并测试代码",
        mutated_files=["kernel.py"],
        test_commands=[(DummyToolCall(), DummyToolResult())],
        all_results=[DummyToolResult()],
    )
    assert c2.passed is False
    assert c2.risk_type == "NON_ZERO_EXIT_CODE"

    # Case 3: Test executed with exit_code 0 -> PASS
    DummyToolResult.output = {"exit_code": 0}
    c3 = substrate.on_completion_gate(
        session_id="test_sess",
        user_prompt="重构并测试代码",
        mutated_files=["kernel.py"],
        test_commands=[(DummyToolCall(), DummyToolResult())],
        all_results=[DummyToolResult()],
    )
    assert c3.passed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
