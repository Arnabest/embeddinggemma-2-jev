# -*- coding: utf-8 -*-
"""CommandGuardDecider: 终端高危命令静态语义审计与破坏力分级."""
from __future__ import annotations

import re
from typing import Dict, Any
from .base import JevDeciderBase


class CommandGuardDecider:
    def __init__(self, base: JevDeciderBase):
        self.base = base

    def audit_command_safety(
        self,
        command_line: str,
        cwd: str = "",
        is_daemon: bool = False
    ) -> Dict[str, Any]:
        """
        审计待执行命令行的物理与语义安全性。
        返回:
        - danger_level: 1 (安全) 到 5 (极度危险/致命破坏)
        - is_blocked: 是否触发硬门禁拦截
        - risk_category: DESTRUCTIVE_DELETE | PROCESS_KILL | SYSTEM_MUTATION | SUSPICIOUS_NETWORK | SAFE
        - reason: 归因解释
        - latency_ms: 推理耗时
        """
        cmd_clean = command_line.strip()
        cmd_lower = cmd_clean.lower()

        # 1. 物理模式特征匹配
        risk_category = "SAFE"
        hard_danger_level = 1
        hard_reason = "命令语义安全，常规只读或正常构建任务。"

        # 危险删除与清盘
        if any(p in cmd_lower for p in (
            "rm -rf", "remove-item -recurse -force", "del /f /s /q", "rd /s /q",
            "shutil.rmtree", "format ", "diskpart", "drop table", "drop database"
        )):
            risk_category = "DESTRUCTIVE_DELETE"
            hard_danger_level = 5
            hard_reason = "检测到破坏性递归强制删除或磁盘格式化特征。"

        # 强杀进程
        elif any(p in cmd_lower for p in (
            "kill -9", "taskkill /f", "stop-process -force", "pkill -9"
        )):
            risk_category = "PROCESS_KILL"
            hard_danger_level = 4
            hard_reason = "检测到强制击杀进程命令，可能导致数据丢失或端口意外断开。"

        # 系统底盘与注册表变异
        elif any(p in cmd_lower for p in (
            "reg add", "reg delete", "bcdedit", "net user", "chmod 777"
        )):
            risk_category = "SYSTEM_MUTATION"
            hard_danger_level = 4
            hard_reason = "检测到系统关键配置、注册表或最高权限修改操作。"

        # 2. 结合 Jev 因果裁决
        options = [
            {"t": "SAFE", "d": "常规开发、单测执行、代码检索或只读诊断命令 (Level 1-2)"},
            {"t": "MODERATE", "d": "常规包安装、文件写入或常规编译构建任务 (Level 3)"},
            {"t": "HIGH_RISK", "d": "进程强杀、递归删除、破坏性变更或系统底盘修改 (Level 4-5)"}
        ]
        question = "请对以下终端命令的破坏力与安全性进行评估："
        state = f"命令行: {cmd_clean}\n工作目录: {cwd}\n后台常驻属性: {is_daemon}\n规则预警: {risk_category} (Lvl {hard_danger_level})"

        dec = self.base.jev_decide("choice", question, options, state)
        jev_choice = options[dec.get("pred", 0)]["t"]

        danger_level = hard_danger_level
        if hard_danger_level == 1:
            if jev_choice == "HIGH_RISK":
                danger_level = 4
                risk_category = "HIGH_RISK"
                hard_reason = "Jev 语义判定当前命令存在潜在高危或不可逆破坏性风险。"
            elif jev_choice == "MODERATE":
                danger_level = 3
                risk_category = "MODERATE"
                hard_reason = "常规构建或状态变更操作。"

        is_blocked = danger_level >= 4

        return {
            "command": cmd_clean,
            "danger_level": danger_level,
            "is_blocked": is_blocked,
            "risk_category": risk_category,
            "reason": hard_reason,
            "confidence": dec.get("conf", 0.0),
            "latency_ms": dec.get("ms", 0.0),
            "device": dec.get("device", "unknown"),
            "recommendation": (
                "高危破坏性操作，建议在 Agent Loop 中触发用户二次确认 (decision: ask)。" if is_blocked
                else "放行执行 (decision: allow)。"
            )
        }
