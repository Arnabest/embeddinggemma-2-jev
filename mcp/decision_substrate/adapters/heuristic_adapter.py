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


class HeuristicDecisionAdapter(BaseDecisionModelAdapter):
    """Deterministic, zero-latency symbolic rule engine."""

    @property
    def model_name(self) -> str:
        return "heuristic_symbolic_engine"

    def _extract_slots(self, prompt: str) -> Dict[str, Any]:
        """轻量级槽位抽取器 (借鉴 Schema-Guided Slot Filling 理念)."""
        slots: Dict[str, Any] = {}
        # 1. URL 提取
        urls = re.findall(r'https?://[^\s<>"]+', prompt)
        if urls:
            slots["urls"] = urls
        # 2. 文件路径模式
        files = re.findall(r'[\w\-./\\]+\.(?:py|md|json|ts|tsx|js|html|bat|sh|yaml|yml)', prompt)
        if files:
            slots["target_files"] = list(set(files))
        # 3. Git 分支/引用
        git_match = re.search(r'git\s+(?:checkout|reset|revert|switch|branch)\s+([a-zA-Z0-9_\-./]+)', prompt)
        if git_match:
            slots["git_target"] = git_match.group(1)
        return slots

    def infer_intent(self, prompt: str, context: str = "") -> IntentDecision:
        p_lower = prompt.lower()
        c_lower = context.lower()
        combined = f"{p_lower} {c_lower}"
        slots = self._extract_slots(prompt)

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

        # 3. Browser Action & CDP Sensing (网页嗅探与前端交互)
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
        if any(w in p_lower for w in ("查看", "解释", "阅读", "查询", "什么是", "怎么用", "explain", "read", "view", "how to", "search", "文档", "doc")):
            return IntentDecision(
                intent="READONLY_QUERY",
                confidence=0.88,
                recommended_skills=["model-catalog"],
                risk_level="LOW",
                description="Read-only query, document inspection, or explanation",
                slots=slots,
            )

        # 9. Code Mutate (Default for software engineering)
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
        # Read-only tools are always allowed
        if tool_name in ("view_file", "search_web", "call_mcp_tool", "vault_search", "shelf_search", "inspect_trajectory"):
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

        if tool_name in ("write_to_file", "replace_file_content"):
            target = str(args.get("TargetFile", "")).lower()
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
