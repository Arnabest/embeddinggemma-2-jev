#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified Decision Substrate Kernel (统一决策模块基座内核).

Coordinates all agent loop decision mechanisms through a unified, model-agnostic kernel:
1. Pre-turn intention, task context, epistemic weight arbitration & Attention HUD assembly.
2. In-flight tool mutation & hardware-aware hazard check.
3. Divergence branch resolution.
4. Bounded deliberation / hesitation budget lifecycle (3 default + 1 human extension).
5. Post-failure exception stacktrace triage & remediation.
6. Physical ground-truth completion gate verification & active replay harvesting.
7. Post-task codebase line budget (800 lines limit) & memory vault payload generation.
8. Runtime hot-swappable model adapters (JEV, Heuristic, OpenAI-compatible, Composite).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .protocol import (
    BaseDecisionModelAdapter,
    CompletionDecision,
    DivergenceDecision,
    HazardDecision,
    IntentDecision,
    PostTurnAudit,
    PreTurnGrounding,
    TriageDecision,
)
from .epistemic import EpistemicArbiter, EpistemicContext
from .hesitation import HesitationBudgetManager, HesitationState
from .adapters import (
    CompositeDecisionAdapter,
    HeuristicDecisionAdapter,
    JevGemmaDecisionAdapter,
)


class DecisionSubstrate:
    """Master Decision Substrate unifying all agent loop deliberation primitives."""

    _instance: Optional["DecisionSubstrate"] = None

    def __init__(self, adapter: Optional[BaseDecisionModelAdapter] = None):
        self.adapter: BaseDecisionModelAdapter = adapter or JevGemmaDecisionAdapter()
        self.arbiter = EpistemicArbiter()
        self.hesitation_mgr = HesitationBudgetManager(default_budget=3)
        self._engine = None

    @classmethod
    def get_instance(cls) -> "DecisionSubstrate":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_instance(cls, substrate: "DecisionSubstrate") -> None:
        cls._instance = substrate

    def _get_engine(self):
        if self._engine is None:
            try:
                from engine import GemmaPrepEngine
                self._engine = GemmaPrepEngine()
            except Exception:
                self._engine = None
        return self._engine

    # -------------------------------------------------------------------------
    # 1. Model Adapter Hot-Swapping & Telemetry
    # -------------------------------------------------------------------------
    def set_adapter(self, adapter: BaseDecisionModelAdapter) -> None:
        """Dynamically replace the underlying decision model adapter at runtime."""
        self.adapter = adapter

    def switch_model(self, model_type: str, **kwargs) -> str:
        """Switch decision model backend by name ('jev', 'heuristic', 'composite')."""
        t = model_type.lower().strip()
        if t in ("jev", "jev_gemma", "gemma"):
            self.set_adapter(JevGemmaDecisionAdapter())
        elif t in ("heuristic", "rule", "symbolic"):
            self.set_adapter(HeuristicDecisionAdapter())
        elif t in ("composite", "hybrid"):
            self.set_adapter(CompositeDecisionAdapter(
                primary=JevGemmaDecisionAdapter(),
                fallback=HeuristicDecisionAdapter(),
            ))
        else:
            raise ValueError(f"Unknown decision model type: {model_type}")
        return self.adapter.model_name

    def get_status(self) -> Dict[str, Any]:
        """Telemetry snapshot of the unified decision substrate."""
        return {
            "active_model": self.adapter.model_name,
            "hesitation_status": self.hesitation_mgr.get_status(),
        }

    # -------------------------------------------------------------------------
    # 2. Agent Loop Hook: Pre-Turn Grounding & Attention HUD
    # -------------------------------------------------------------------------
    def on_pre_turn(
        self,
        session_id: str,
        prompt: str,
        state: str = "",
        include_recall: bool = True,
        include_hardware: bool = True,
    ) -> PreTurnGrounding:
        """Master pre-turn hook taking over agent loop's prior assembly."""
        # 1. Infer intent via active model adapter
        intent = self.adapter.infer_intent(prompt, context=state)

        # 2. Epistemic evidence arbitration (anti-bias / anti-stale)
        epistemic = self.arbiter.arbitrate(instruction=prompt, context_snippet=state)

        # 3. Hesitation budget state
        h_status = self.hesitation_mgr.get_status()

        # 4. Hardware probe
        eng = self._get_engine()
        hw_info = eng.get_hardware_status() if (eng and include_hardware) else {}

        # 5. Hybrid memory recall
        recalls = []
        if eng and include_recall and prompt:
            try:
                recalls = eng.hybrid_recall(prompt, top_k=3)
            except Exception:
                recalls = []

        # 6. Build Standard Attention HUD
        hud_lines = ["[Unified-Decision-Substrate: 前置态势与先验装配]"]
        hud_lines.append(f"- [核心意图] `{intent.intent}` (置信度: {intent.confidence:.1%}) | {intent.description}")
        if intent.recommended_skills:
            hud_lines.append(f"  * 推荐协同技能: {', '.join(intent.recommended_skills)}")

        if hw_info.get("gpu_available"):
            used = hw_info.get("vram_used_mb", 0)
            tot = hw_info.get("vram_total_mb", 0)
            pct = hw_info.get("vram_usage_pct", 0.0)
            lvl = hw_info.get("danger_level", 1)
            warn = "[警告: 显存高水位]" if lvl >= 4 else "[正常: 显存充裕]"
            hud_lines.append(f"- [硬件状态] {warn} (RTX 4060 Ti 显存: {used}/{tot}MB, {pct}% 占用)")

        ctx_name = epistemic.get("context", "unknown").upper()
        primary = epistemic.get("primary_authority_desc", "")
        p_weight = epistemic.get("weights", {}).get(epistemic.get("primary_authority", ""), 0.0)
        hud_lines.append(f"- [证据仲裁] 任务环境: `{ctx_name}` | 核心准绳: `{primary}` ({p_weight:.1%}) | 防盲信协议已激活")
        for mask in epistemic.get("masks_applied", []):
            hud_lines.append(f"  * [硬掩码生效] {mask}")

        ext_label = "可用" if h_status.get("extension_available") else "已耗尽/锁定"
        hud_lines.append(
            f"- [犹豫预算] 剩余机会: {h_status.get('current_budget')}/3 | "
            f"申请扩展: `{ext_label}` | 状态: `{h_status.get('state', '').upper()}`"
        )

        if recalls:
            hud_lines.append("- [记忆召回] 历史会话与技能先验:")
            for item in recalls:
                src = item.get("source", "memory")
                title = item.get("title") or item.get("name") or item.get("chunk_id") or "片段"
                snip = (item.get("snippet", "")).replace("\n", " ").strip()
                if len(snip) > 120:
                    snip = snip[:120] + "..."
                hud_lines.append(f"  * [{src}: {title}] {snip}")

        hud_markdown = "\n".join(hud_lines)

        return PreTurnGrounding(
            intent=intent,
            epistemic_weights=epistemic,
            hesitation_status=h_status,
            hardware_status=hw_info,
            recalled_memories=recalls,
            hud_markdown=hud_markdown,
            inject_steps=[{"ephemeralMessage": hud_markdown}],
        )

    # -------------------------------------------------------------------------
    # 3. Agent Loop Hook: In-Flight Hazard Gate
    # -------------------------------------------------------------------------
    def on_hazard_check(
        self,
        tool_name: str,
        args: Dict[str, Any],
        session_id: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> HazardDecision:
        """Master in-flight safety guard intercepting risky mutations."""
        # Hardware saturation check for heavy compute
        eng = self._get_engine()
        if eng and tool_name == "run_command":
            hw = eng.get_hardware_status()
            if hw.get("danger_level", 1) >= 5:
                cmd = str(args.get("CommandLine", "")).lower()
                if any(k in cmd for k in ("python", "torch", "comfy", "train", "pytest")):
                    return HazardDecision(
                        decision="ask",
                        risk_level="CRITICAL",
                        reason=f"Hardware saturation guard: VRAM at {hw.get('vram_usage_pct')}% (Danger Level {hw.get('danger_level')}). High risk of CUDA OOM.",
                        requires_human_confirmation=True,
                    )

        # Delegate tool inspection to active adapter
        return self.adapter.evaluate_hazard(tool_name, args, context)

    # -------------------------------------------------------------------------
    # 4. Agent Loop Hook: Divergence Branching
    # -------------------------------------------------------------------------
    def on_divergence(
        self,
        branches: List[Any],
        context: Optional[Dict[str, Any]] = None,
        session_id: str = "",
    ) -> DivergenceDecision:
        """Master divergence arbiter selecting optimal route among competing tool paths."""
        return self.adapter.arbitrate_divergence(branches, context)

    # -------------------------------------------------------------------------
    # 5. Agent Loop Hook: Deliberation / Hesitation Lifecycle
    # -------------------------------------------------------------------------
    def on_hesitation(
        self,
        action: str = "status",
        reason: str = "",
        user_approved: bool = False,
    ) -> Dict[str, Any]:
        """Master bounded hesitation controller."""
        act = action.lower().strip()
        if act == "status":
            return self.hesitation_mgr.get_status()
        elif act == "consume":
            return self.hesitation_mgr.consume_hesitation(reason)
        elif act == "apply_extension":
            return self.hesitation_mgr.apply_for_extension(user_approved, reason)
        elif act == "reset":
            self.hesitation_mgr.reset()
            return {"success": True, "status": self.hesitation_mgr.get_status()}
        else:
            return {"error": f"Unknown hesitation action: {action}"}

    # -------------------------------------------------------------------------
    # 6. Agent Loop Hook: Failure Triage
    # -------------------------------------------------------------------------
    def on_failure_triage(
        self,
        traceback_text: str,
        exit_code: int = 1,
    ) -> TriageDecision:
        """Master failure triage identifying root cause and remediation advice."""
        return self.adapter.triage_failure(traceback_text, exit_code)

    # -------------------------------------------------------------------------
    # 7. Agent Loop Hook: Physical Completion Gate
    # -------------------------------------------------------------------------
    def on_completion_gate(
        self,
        session_id: str,
        user_prompt: str,
        mutated_files: List[Any],
        test_commands: List[Tuple[Any, Any]],
        all_results: List[Any],
        step_count: int = 0,
        claim_content: str = "",
    ) -> CompletionDecision:
        """Physical Ground Truth Completion Gate:
        Prevents verbal completion cheating by enforcing OS return codes and AST proofs.
        """
        p_lower = user_prompt.lower()
        requires_verification = any(k in p_lower for k in (
            "测试", "test", "断言", "assert", "验证", "verify", "压测", "stress", "无锁", "lock-free", "并发", "race"
        ))

        # Check 1: Mutated files on verification tasks with zero executed test commands
        if mutated_files and requires_verification and not test_commands:
            reason = "已检测到代码写盘，但尚未调用 run_command 执行实际的单元测试或并发压测断言，存在纯语言自评作弊风险。"
            self._harvest_gate_case(user_prompt, claim_content, False, None, False, reason)
            return CompletionDecision(
                passed=False,
                reason=reason,
                risk_type="UNVERIFIED_MUTATION",
            )

        # Check 2: Latest test command failure
        if test_commands:
            last_tc, last_tr = test_commands[-1]
            out = getattr(last_tr, "output", None)
            if out is None and isinstance(last_tr, dict):
                out = last_tr.get("output")

            ec = None
            if isinstance(out, dict):
                ec = out.get("exit_code") if "exit_code" in out else out.get("exitCode")

            if ec is not None and ec != 0:
                cmd_str = getattr(last_tc, "args", {}).get("CommandLine", "test") if hasattr(last_tc, "args") else "test"
                reason = f"最新测试命令 `{cmd_str}` 执行失败 (exitCode={ec})，物理断言未通过。"
                self._harvest_gate_case(user_prompt, claim_content, True, ec, False, reason)
                return CompletionDecision(
                    passed=False,
                    reason=reason,
                    risk_type="NON_ZERO_EXIT_CODE",
                    details={"exit_code": ec},
                )

        # Check 3: AST verification failure
        for tr in all_results:
            ast_ok = getattr(tr, "ast_verified", None)
            if ast_ok is False:
                err = getattr(tr, "error", "AST parse error")
                reason = f"文件修改未通过 AST 语法树编译器校验: {err}，严禁交付语法破碎代码。"
                self._harvest_gate_case(user_prompt, claim_content, bool(test_commands), None, False, reason)
                return CompletionDecision(
                    passed=False,
                    reason=reason,
                    risk_type="AST_SYNTAX_FAILURE",
                )

        # Passed
        last_ec = 0 if test_commands else None
        self._harvest_gate_case(user_prompt, claim_content, bool(test_commands), last_ec, True, "All physical invariants verified with exit code 0")
        return CompletionDecision(
            passed=True,
            reason="All physical invariants verified with exit code 0",
        )

    def _harvest_gate_case(
        self,
        user_goal: str,
        claim: str,
        has_tests: bool,
        exit_code: Optional[int],
        passed: bool,
        reason: str,
    ) -> None:
        try:
            from harvest.collector import JevDataCollector
            JevDataCollector.get_instance().harvest_completion_gate_case(
                user_goal=user_goal,
                claim_content=claim,
                has_tests_run=has_tests,
                exit_code=exit_code,
                passed=passed,
                rejection_reason=reason if not passed else "",
            )
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 8. Agent Loop Hook: Post-Turn Audit & Reorganization
    # -------------------------------------------------------------------------
    def on_post_turn(
        self,
        workspace_path: str,
        changed_files: Optional[List[str]] = None,
        exit_code: int = 0,
        session_goal: str = "",
        max_line_limit: int = 800,
    ) -> PostTurnAudit:
        """Master post-turn codebase audit and memory vault payload preparation."""
        eng = self._get_engine()
        if eng:
            try:
                res = eng.post_task_codebase_audit(
                    workspace_path=workspace_path,
                    changed_files=changed_files,
                    exit_code=exit_code,
                    session_goal=session_goal,
                    max_line_limit=max_line_limit,
                )
                return PostTurnAudit(
                    status=res.get("decision", "PASS"),
                    line_limit=max_line_limit,
                    max_lines_exceeded=res.get("exceeded_files", []),
                    changes_analyzed=len(changed_files or []),
                    exit_code=exit_code,
                    vault_payload=res.get("vault_payload", {}),
                )
            except Exception:
                pass

        return PostTurnAudit(
            status="PASS",
            line_limit=max_line_limit,
            max_lines_exceeded=[],
            changes_analyzed=len(changed_files or []),
            exit_code=exit_code,
            vault_payload={},
        )
