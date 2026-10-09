# -*- coding: utf-8 -*-
"""FailureTriageDecider: 报错堆栈与单测失败秒级归因分流器."""
from __future__ import annotations

import re
from typing import Dict, Any, List
from .base import JevDeciderBase


class FailureTriageDecider:
    def __init__(self, base: JevDeciderBase):
        self.base = base

    def triage_failure(
        self,
        traceback_text: str,
        exit_code: int = 1
    ) -> Dict[str, Any]:
        """
        对程序崩溃或测试失败堆栈进行毫秒级归因分类，生成行动补丁建议。
        返回:
        - failure_category: DEPENDENCY_MISSING | SYNTAX_IMPORT_CYCLE | ASSERTION_LOGIC | TIMEOUT_DEADLOCK | IO_PERMISSION | RUNTIME_CRASH
        - confidence: 置信度
        - offending_line: 核心引发报错行
        - remediation_capsule: 针对性排错微胶囊
        """
        tb_clean = traceback_text.strip()
        lines = [l.strip() for l in tb_clean.splitlines() if l.strip()]

        # 1. 提取尾部核心报错行
        offending_line = lines[-1] if lines else "Exit Code Non-Zero"
        for line in reversed(lines):
            if any(k in line for k in ("Error:", "Exception:", "FAILED", "fatal:", "Traceback")):
                offending_line = line
                break

        tb_lower = tb_clean.lower()

        # 2. 规则先验判定
        category = "RUNTIME_CRASH"
        advice = "检查运行时输入与未捕获异常上下文。"

        if any(k in tb_lower for k in ("modulenotfounderror", "importerror: no module named", "dll load failed")):
            category = "DEPENDENCY_MISSING"
            advice = "本地环境缺失对应第三方依赖包，建议先通过 py -3 -m pip install 补齐。"
        elif any(k in tb_lower for k in ("syntaxerror", "indentationerror", "cannot import name", "circular import")):
            category = "SYNTAX_IMPORT_CYCLE"
            advice = "代码存在语法结构损坏或模块循环引用，检查 AST 闭合与 import 引用顺序。"
        elif any(k in tb_lower for k in ("assertionerror", "failed (failures=", "assert ")):
            category = "ASSERTION_LOGIC"
            advice = "物理测试断言失败，对比实际输出值与预期期望，调整核心业务算法实现。"
        elif any(k in tb_lower for k in ("timeouterror", "timed out", "deadlock", "waitmsbeforeasync")):
            category = "TIMEOUT_DEADLOCK"
            advice = "进程等待超时或标准管道死锁，排查并发锁竞争或非阻塞异步读写机制。"
        elif any(k in tb_lower for k in ("permissionerror", "filenotfounderror", "access is denied")):
            category = "IO_PERMISSION"
            advice = "文件路径不存在或被操作系统占用，核验绝对路径合法性与文件独占句柄。"

        # 3. Jev 因果校验与分类
        options = [
            {"t": "DEPENDENCY_MISSING", "d": "缺失 Python 模块或动态链接库"},
            {"t": "SYNTAX_IMPORT_CYCLE", "d": "语法错误、缩进错误或包循环导入"},
            {"t": "ASSERTION_LOGIC", "d": "单测断言失败或业务逻辑返回值不符"},
            {"t": "TIMEOUT_DEADLOCK", "d": "子进程管道阻塞、锁死或执行超时"},
            {"t": "IO_PERMISSION", "d": "物理路径缺失或文件权限拒绝"},
            {"t": "RUNTIME_CRASH", "d": "常规运行时未捕获异常或环境崩溃"}
        ]
        question = "请对以下错误堆栈的根本故障原因进行分类："
        state = f"退出码: {exit_code}\n核心报错行: {offending_line}\n规则初步判定: {category}"

        dec = self.base.jev_decide("choice", question, options, state)
        conf = dec.get("conf", 0.0)

        return {
            "exit_code": exit_code,
            "failure_category": category,
            "confidence": conf,
            "offending_line": offending_line,
            "remediation_capsule": advice,
            "latency_ms": dec.get("ms", 0.0),
            "device": dec.get("device", "unknown")
        }
