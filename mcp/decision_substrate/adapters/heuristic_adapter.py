#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Heuristic Decision Model Adapter (零延迟符号规则适配器).

Pure Python, zero-dependency, deterministic rule and regex engine.
Guarantees 100% offline availability for local agent loop execution.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..protocol import (
    BaseDecisionModelAdapter,
    DivergenceDecision,
    HazardDecision,
    IntentDecision,
    TriageDecision,
)
from deciders.directional_gate import (
    clean_user_text,
    evaluate_directional_intent,
    evaluate_directional_tool_gate,
    extract_slots,
    is_readonly_query,
    is_readonly_tool_call,
)


class HeuristicDecisionAdapter(BaseDecisionModelAdapter):
    """Deterministic, zero-latency symbolic rule engine."""

    @property
    def model_name(self) -> str:
        return "heuristic_symbolic_engine"

    def _extract_slots(self, prompt: str) -> Dict[str, Any]:
        """轻量级槽位抽取器 (委托给共享 directional_gate)."""
        return extract_slots(prompt)

    def infer_intent(self, prompt: str, context: str = "", session_mode: str = "") -> IntentDecision:
        clean_prompt = clean_user_text(prompt)
        p_lower = clean_prompt.lower()
        c_lower = context.lower()
        combined = f"{p_lower} {c_lower}"
        slots = self._extract_slots(clean_prompt)

        # 1. High Hazard Checks
        if any(w in combined for w in ("rmdir /s", "rm -rf", "drop database", "format ", "mkfs", "delete from", "清空磁盘", "格式化")):
            return IntentDecision(
                intent="HIGH_HAZARD",
                confidence=0.99,
                recommended_skills=["pre-action-audit"],
                risk_level="CRITICAL",
                description="High destructive command or file mutation detected",
                slots=slots,
            )

        # 2. State Rollback & Undo (状态守恒与防御性回滚)
        if any(w in p_lower for w in ("git reset", "git revert", "git checkout .", "git clean", "回滚", "撤销修改", "还原快照", "rollback", "undo", "消除脏改动")):
            return IntentDecision(
                intent="STATE_ROLLBACK",
                confidence=0.92,
                recommended_skills=["nexus-scaffold"],
                risk_level="LOW",
                description="Safe state recovery, Git shadow rollback or dirty-diff purge",
                slots=slots,
            )

        # 3. Directional Planning Pre-Alignment (方向性指令前置交流与规划门禁)
        dir_eval = evaluate_directional_intent(clean_prompt, context=context, slots=slots)
        if dir_eval["is_directional"]:
            return IntentDecision(
                intent="DIRECTIONAL_PLANNING",
                confidence=dir_eval["confidence"],
                recommended_skills=["pre-action-audit"],
                risk_level="LOW",
                description="宏观方向性指令，须先只读勘察并与用户交流细节及规划要求后再执行",
                slots=slots,
                raw=dir_eval,
                requires_clarification=True,
            )

        # 4. Browser Action & CDP Sensing (网页嗅探与前端交互)
        if any(w in p_lower for w in ("browser", "cdp", "网页", "嗅探", "sniffer", "bilibili", "b站", "弹幕", "三连", "油猴", "tampermonkey", "tab_id")):
            return IntentDecision(
                intent="BROWSER_ACTION",
                confidence=0.91,
                recommended_skills=["browser-sniffer", "bilibili-comment-action", "bilibili-triple-action"],
                risk_level="LOW",
                description="Chrome CDP network sniffing, DOM perception, or Web automation",
                slots=slots,
            )

        # 4. Multimodal Process (语音合成、转写与OCR)
        if any(w in p_lower for w in ("语音合成", "tts", "gpt-sovits", "sovits", "音频转写", "字幕", "funasr", "asr", "ocr", "提取文字", "mineru")):
            return IntentDecision(
                intent="MULTIMODAL_PROCESS",
                confidence=0.90,
                recommended_skills=["gpt-sovits-synthesize", "media-transcribe", "document-ocr"],
                risk_level="LOW",
                description="Speech synthesis (TTS), audio transcription (ASR), or document OCR",
                slots=slots,
            )

        # 5. Swarm Orchestration (子代理并发与蜂群编排)
        if any(w in p_lower for w in ("子代理", "subagent", "并发探索", "蜂群", "swarm", "多路径推理", "对立探索", "并行执行", "fork_agent")):
            return IntentDecision(
                intent="SWARM_ORCHESTRATE",
                confidence=0.89,
                recommended_skills=["nexus-scaffold"],
                risk_level="LOW",
                description="Multi-path reasoning rollout, subagent branching, or swarm orchestration",
                slots=slots,
            )

        # 6. Deep Research (学术论文研读与算法提炼)
        if any(w in p_lower for w in ("论文", "arxiv", "公式推导", "消融实验", "paper", "研究报告", "研报总结", "算法提炼")):
            return IntentDecision(
                intent="DEEP_RESEARCH",
                confidence=0.88,
                recommended_skills=["paper-distiller", "media-to-report"],
                risk_level="LOW",
                description="In-depth academic paper distillation, math formula deduction, or ablation analysis",
                slots=slots,
            )

        # 7. System Command / Diagnostics
        if any(w in p_lower for w in ("run_command", "pip install", "powershell", "curl ", "netstat", "ps aux", "docker ", "重启服务", "运行命令", "执行脚本")):
            return IntentDecision(
                intent="SYSTEM_COMMAND",
                confidence=0.90,
                recommended_skills=["devtools-diagnostics"],
                risk_level="MEDIUM",
                description="Terminal system execution or diagnostic request",
                slots=slots,
            )

        # 8. Readonly Query
        if is_readonly_query(clean_prompt, context):
            return IntentDecision(
                intent="READONLY_QUERY",
                confidence=0.88,
                recommended_skills=["model-catalog"],
                risk_level="LOW",
                description="Read-only query, document inspection, or explanation",
                slots=slots,
            )

        # 9. Session-aware default fallback (standalone Q&A/scratch vs software engineering)
        eff_mode = session_mode
        if not eff_mode:
            m_mode = re.search(r"\[SESSION_MODE:(workspace_bound|standalone_targeted_repo|standalone_scratch|standalone_chat_research)\]", context or "")
            if m_mode:
                eff_mode = m_mode.group(1)
        if eff_mode in ("standalone_chat_research", "standalone_scratch"):
            from deciders.session_context import has_explicit_code_mutation_signal

            if not has_explicit_code_mutation_signal(clean_prompt):
                return IntentDecision(
                    intent="READONLY_QUERY",
                    confidence=0.78,
                    recommended_skills=["model-catalog"],
                    risk_level="LOW",
                    description="Standalone Q&A / research or scratch exploration without project repo mutation",
                    slots=slots,
                )

        # 10. Code Mutate (Default for software engineering)
        return IntentDecision(
            intent="CODE_MUTATE",
            confidence=0.85,
            recommended_skills=["nexus-scaffold", "python-modularizer"],
            risk_level="MEDIUM",
            description="Software code modification, refactoring, bugfix, or test creation",
            slots=slots,
        )

    def evaluate_hazard(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> HazardDecision:
        # Directional planning lock gate (must confirm details before mutation/execution)
        if context and context.get("directional_locked"):
            gate = evaluate_directional_tool_gate(tool_name, args, locked=True)
            if gate:
                return HazardDecision(
                    decision="deny",
                    risk_level="HIGH",
                    reason=gate["reason"],
                    is_blocked=True,
                )

        # Read-only tools are always allowed
        if is_readonly_tool_call(tool_name, args) or tool_name in ("inspect_trajectory",):
            return HazardDecision(decision="allow", risk_level="LOW")

        # Mutating operations
        if tool_name == "run_command":
            cmd = str(args.get("CommandLine", "")).lower()
            if any(k in cmd for k in ("rm -rf /", "rmdir /s /q c:\\", "format ", "mkfs", "drop database")):
                return HazardDecision(
                    decision="deny",
                    risk_level="CRITICAL",
                    reason=f"Destructive system command intercepted: {cmd}",
                    is_blocked=True,
                )
            if any(k in cmd for k in ("git reset --hard", "git clean -fd", "drop table", "shutdown")):
                return HazardDecision(
                    decision="ask",
                    risk_level="HIGH",
                    reason=f"Potentially irreversible command requires human confirmation: {cmd}",
                    requires_human_confirmation=True,
                )

        if tool_name in ("write_file", "edit_file", "write_to_file", "replace_file_content", "multi_replace_file_content"):
            target = str(args.get("Path") or args.get("TargetFile") or "").lower()
            if any(p in target for p in ("c:\\windows", "c:\\program files", "/etc", "/bin", "/boot")):
                return HazardDecision(
                    decision="deny",
                    risk_level="CRITICAL",
                    reason=f"System directory write access prohibited: {target}",
                    is_blocked=True,
                )
            if any(p in target for p in (".git/", ".git\\")):
                return HazardDecision(
                    decision="deny",
                    risk_level="CRITICAL",
                    reason=f"Git metadata write access prohibited: {target}",
                    is_blocked=True,
                )

        return HazardDecision(decision="allow", risk_level="LOW")

    def triage_failure(
        self,
        traceback_text: str,
        exit_code: int = 1,
    ) -> TriageDecision:
        tb = traceback_text.lower()
        if "modulenotfounderror" in tb or "no module named" in tb:
            match = re.search(r"no module named ['\"]([^'\"]+)['\"]", tb)
            mod = match.group(1) if match else "unknown"
            return TriageDecision(
                primary_cause=f"Missing Python Dependency: {mod}",
                component="python_runtime",
                confidence=0.95,
                suggested_fix=f"Install missing dependency via pip: python -m pip install {mod}",
            )
        if "assertionerror" in tb or "assert " in tb:
            return TriageDecision(
                primary_cause="Unit Test Assertion Failure",
                component="test_suite",
                confidence=0.92,
                suggested_fix="Inspect expected vs actual values in test assertion and align implementation logic.",
            )
        if "syntaxerror" in tb or "indentationerror" in tb:
            return TriageDecision(
                primary_cause="Python Syntax / Indentation Error",
                component="parser",
                confidence=0.98,
                suggested_fix="Verify AST compliance and fix indentation/syntax tokens before running.",
            )
        if "permissionerror" in tb or "access is denied" in tb:
            return TriageDecision(
                primary_cause="Filesystem Permission Denied",
                component="os_filesystem",
                confidence=0.90,
                suggested_fix="Check file lock, close existing handles or run with elevated permissions.",
            )

        return TriageDecision(
            primary_cause=f"General Execution Failure (exit_code={exit_code})",
            component="execution_engine",
            confidence=0.70,
            suggested_fix="Inspect standard error output and diagnose specific faulting line.",
        )

    def arbitrate_divergence(
        self,
        branches: List[Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> DivergenceDecision:
        if not branches:
            return DivergenceDecision(chosen_branch="", rationale="No branches provided", confidence=1.0)
        str_branches = [str(b) for b in branches]
        for b in str_branches:
            if any(k in b.lower() for k in ("read", "view", "get", "status", "inspect", "search")):
                return DivergenceDecision(
                    chosen_branch=b,
                    rationale="Selected safe observation branch before executing mutations",
                    confidence=0.85,
                    discarded_branches=[x for x in str_branches if x != b],
                )
        return DivergenceDecision(
            chosen_branch=str_branches[0],
            rationale="Selected primary sequential execution path",
            confidence=0.80,
            discarded_branches=str_branches[1:],
        )
