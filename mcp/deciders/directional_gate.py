#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Directional Planning Gate & Context JEV Pre-Alignment Decider.

Provides deterministic + JEV-aligned classification and enforcement for
directional / macro instructions that require alignment on details and
planning before any code mutation or command execution begins.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_LOCK_STORE = Path(__file__).resolve().parent.parent / "data" / "directional_locks.json"

READONLY_TOOLS = frozenset({
    "read_file",
    "view_file",
    "grep_search",
    "find_by_name",
    "list_dir",
    "search_web",
    "read_url_content",
    "ask_question",
    "vault_search",
    "vault_read",
    "vault_list",
    "vault_stats",
    "shelf_search",
    "shelf_read",
    "shelf_catalog",
    "list_resources",
    "read_resource",
    "list_plugin_accounts",
    "search_marketplace",
    "manage_task",
    "manage_subagents",
})

READONLY_MCP_TOOLS = frozenset({
    "vault_search",
    "vault_read",
    "vault_list",
    "vault_stats",
    "shelf_search",
    "shelf_read",
    "shelf_catalog",
    "quota_inspect",
    "gemma_intent_classify",
    "gemma_memory_recall",
    "gemma_jev_decide",
    "gemma_assemble_context",
    "gemma_hardware_status",
    "gemma_model_route",
    "gemma_dependency_route",
    "gemma_evolution_status",
    "book_search",
    "book_list_categories",
    "book_get_detail",
    "book_stats",
    "book_get_graph",
    "model_list_providers",
    "model_query_catalog",
    "model_get_parameter_recommendation",
    "model_pull_live_status",
    "model_stats",
    "frontend_search_lib",
    "frontend_get_recipe",
    "frontend_list_catalog",
    "sniffer_list_tabs",
    "sniffer_get_current_view",
    "sniffer_query_traffic",
    "sniffer_get_detail",
    "sniffer_get_console_logs",
    "sidebar_list_tabs",
    "sidebar_get_view",
})

MUTATING_OR_EXEC_TOOLS = frozenset({
    "write_file",
    "edit_file",
    "write_to_file",
    "replace_file_content",
    "multi_replace_file_content",
    "nexus_patch_chunk",
    "run_command",
})

SHORT_CONFIRMATIONS = frozenset({
    "可以", "好的", "行", "收到", "确认", "继续", "明白", "同意",
    "开始", "开始吧", "直接开始", "直接动手", "按方案", "按这个",
    "ok", "okay", "yes", "sure", "proceed", "go ahead", "fine", "y",
    "approved", "lgtm", "do it",
})

_HOLD_OFF_CUES = (
    "先别动手", "不要马上动手", "先别改", "先别执行", "先交流细节",
    "先列规划", "先列出规划", "再细化", "再讨论", "换个方向", "重新规划",
)

_EXPLICIT_DIRECTIONAL_PATTERNS = (
    r"大方向",
    r"方向性",
    r"(?:往|向).{1,24}方向",
    r"路线图|roadmap",
    r"总体方案|整体方案|总体规划|整体规划|详细规划",
    r"架构设计|顶层设计|整体设计|系统蓝图|架构演进",
    r"整体重构|全面重构|重新设计|redesign",
    r"重新.{0,12}(?:设计|规划|梳理|改造|架构)",
    r"按用途.{0,10}(?:设计|规划|改造)",
    r"(?:不能|不要)照搬",
    r"弱点补全|系统性(?:补全|改造|梳理)|整体(?:梳理|对齐)",
    r"从头搭建|从零搭建",
)

_MACRO_FRAMING_PREFIXES = (
    "我想做", "我想开发", "我想搭建", "我想构建", "我想打造", "我想实现一个",
    "我想把", "我想给", "我想重构", "我想优化", "我想升级", "我想迁移",
    "打算做", "准备做", "计划做", "想往",
    "帮我做一个", "帮我做个", "帮我搭建", "帮我构建", "帮我设计", "帮我规划", "帮我重构",
    "帮我重构一下", "优化一下", "改善一下", "梳理一下", "重构一下", "改造一下", "升级一下",
    "扩展一下", "重塑", "迁移到",
    "i want to build", "let's build", "let's design", "let's refactor",
    "we should move toward", "plan a ", "architect a ",
)

_BROAD_SCOPE_NOUNS = (
    "系统", "架构", "平台", "工作流", "模块", "生态", "框架", "工具", "基座",
    "助手", "服务", "前端", "后端", "存储", "数据库", "性能", "体验",
    "流程", "机制", "体系", "方案", "项目", "插件", "权限", "引擎",
    "多智能体", "agent", "workflow", "architecture", "system",
    "platform", "pipeline", "framework",
)

_CONCRETE_EDIT_CUES = (
    "修复", "fix", "改成", "替换为", "替换成", "报错", "traceback",
    "attributeerror", "typeerror", "syntaxerror", "keyerror", "bug",
    "添加参数", "增加单测", "补充测试",
)

_READONLY_QUERY_CUES = (
    "查看", "解释", "阅读", "查询", "什么是", "怎么用", "如何查看",
    "还有什么", "有什么", "哪些", "有哪些", "遗漏", "未保留", "未被",
    "判断是否", "是否形成", "是否支持", "是否覆盖", "有没有",
    "运行情况", "怎么样", "多少", "为什么", "区别", "差异", "是否", "能否", "怎么看",
    "评价", "评估", "评测", "评审", "审视", "对比分析", "总体情况", "整体机制", "分析一下", "总结一下",
    "explain", "read", "view", "how to", "check", "status", "list", "search",
    "what is", "what are", "which", "why", "review", "evaluate", "assess", "summarize",
)

_NON_MUTATION_COMPOUND_RE = re.compile(
    r"(?:修改|改动|变更|重构|更新|运行)(?:整体|总体)?(?:请求|范围|影响|记录|历史|闭环|机制|逻辑|方案|边界|情况|状态|效果|日志|结果|统计|健康|模式|流程|表现|环境|架构|质量|前|后|步骤|节点)"
)

_MUTATION_ACTION_BLOCKERS = (
    "删除", "rmdir", "delete", "覆写", "覆盖", "重构", "修改", "写一个",
    "搭建", "构建", "做一个", "实现", "编写", "修复", "添加", "新增", "执行", "运行",
)


def clean_user_text(raw: str) -> str:
    """Strip Antigravity wrapper and metadata tags from user content."""
    if not raw:
        return ""
    text = re.sub(r"<ADDITIONAL_METADATA>[\s\S]*?</ADDITIONAL_METADATA>", "", raw)
    text = re.sub(r"<USER_SETTINGS_CHANGE>[\s\S]*?</USER_SETTINGS_CHANGE>", "", text)
    text = text.replace("<USER_REQUEST>", "").replace("</USER_REQUEST>", "")
    return text.strip()


def extract_slots(instruction: str) -> Dict[str, Any]:
    """Extract lightweight structural slots (URLs, target files, git refs)."""
    slots: Dict[str, Any] = {}
    urls = re.findall(r'https?://[^\s<>"]+', instruction)
    if urls:
        slots["urls"] = urls
    files = re.findall(r'[\w\-./\\]+\.(?:py|md|json|ts|tsx|js|html|bat|sh|yaml|yml)', instruction)
    if files:
        slots["target_files"] = list(set(files))
    git_match = re.search(r'git\s+(?:checkout|reset|revert|switch|branch)\s+([a-zA-Z0-9_\-./]+)', instruction)
    if git_match:
        slots["git_target"] = git_match.group(1)
    return slots


def is_short_confirmation(text: str) -> bool:
    """Return True when text is a brief execution confirmation or plan approval."""
    clean = clean_user_text(text).lower()
    if not clean:
        return False
    if any(w in clean for w in _HOLD_OFF_CUES):
        return False
    if clean in SHORT_CONFIRMATIONS:
        return True
    if re.match(r"^(?:按方案|按这个|照此|按计划|同意|确认|就按|可以开始|开始执行|开始修改)", clean):
        return True
    if re.match(r"^(?:a1:|1\.)\s*", clean) and len(clean) < 160:
        return True
    return False


def _has_hold_off(text_lower: str) -> bool:
    return any(w in text_lower for w in _HOLD_OFF_CUES)


_UNRESOLVED_UI_REPORT_CUES = (
    "未显示", "没显示", "没有显示", "看不到选项", "未弹出", "没弹出", "没有弹出",
    "选项呢", "卡片呢", "输出中断", "生成中断", "中断了", "被截断", "截断了",
    "没发出来", "没有发出来", "未发出", "重试选项", "重新弹出", "再发一次选项", "重发选项",
    "重发一下", "重新调用",
)
_UNRESOLVED_QUESTION_BACK_CUES = (
    "哪些边界", "什么边界", "哪几项", "哪些问题", "什么选项", "哪几个",
    "要确认什么", "要确认啥", "确认哪几", "没看懂", "再说清楚",
)


def _is_unresolved_clarification_followup(text: str) -> bool:
    """Return True when the user reports missing ask_question UI/interruption or asks back about boundaries."""
    clean = clean_user_text(text)
    if not clean:
        return False
    q_lower = clean.lower()
    if any(w in q_lower for w in _UNRESOLVED_UI_REPORT_CUES):
        return True
    if len(clean) <= 40 and not is_short_confirmation(clean):
        if any(w in q_lower for w in _UNRESOLVED_QUESTION_BACK_CUES):
            return True
    return False


def evaluate_directional_intent(
    instruction: str,
    context: str = "",
    slots: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Determine whether an instruction is directional and requires pre-planning dialogue."""
    clean = clean_user_text(instruction)
    if not clean:
        return {
            "is_directional": False,
            "confidence": 0.9,
            "reason": "Empty instruction",
            "signals": [],
            "clarified_by_context": False,
        }

    q_lower = clean.lower()
    c_lower = (context or "").lower()
    resolved_slots = slots if slots is not None else extract_slots(clean)

    # 1. Explicit hold-off / pivot takes precedence over prior clarification
    if _has_hold_off(q_lower) and ("加入上下文jev" not in q_lower and "加入" not in q_lower):
        return {
            "is_directional": True,
            "confidence": 0.95,
            "reason": "User explicitly requested alignment/planning before execution",
            "signals": ["explicit_hold_off"],
            "clarified_by_context": False,
        }

    # 1.5 Unresolved follow-up after interrupted/missing clarification keeps directional lock active
    if "[directional_followup_locked]" in c_lower:
        return {
            "is_directional": True,
            "confidence": 0.93,
            "reason": "Prior directional clarification remains unresolved after user follow-up",
            "signals": ["unresolved_clarification_followup"],
            "clarified_by_context": False,
        }

    # 2. Prior directional clarification already answered by user
    if "[clarification_resolved]" in c_lower:
        return {
            "is_directional": False,
            "confidence": 0.92,
            "reason": "Prior directional clarification resolved in conversation history",
            "signals": ["clarification_resolved"],
            "clarified_by_context": True,
        }

    # 3. Short confirmations / approvals never lock
    if is_short_confirmation(clean):
        return {
            "is_directional": False,
            "confidence": 0.95,
            "reason": "Execution confirmation detected",
            "signals": ["short_confirmation"],
            "clarified_by_context": False,
        }

    # 4. Concrete specialized non-directional commands (rollback, hazard, CLI, pure queries)
    if any(w in q_lower for w in ("git reset", "git revert", "git checkout .", "git clean", "回滚", "撤销修改", "还原快照", "rollback", "undo")):
        return {
            "is_directional": False,
            "confidence": 0.92,
            "reason": "Concrete rollback intent",
            "signals": [],
            "clarified_by_context": False,
        }
    if any(w in q_lower for w in ("rmdir /s", "rm -rf", "drop database", "format ", "mkfs")):
        return {
            "is_directional": False,
            "confidence": 0.95,
            "reason": "High hazard explicit command",
            "signals": [],
            "clarified_by_context": False,
        }
    if re.match(r"^(?:pytest|python\s+-m|pip\s+install|git\s+(?:status|diff|log)|dir|ls|nvidia-smi)\b", q_lower):
        return {
            "is_directional": False,
            "confidence": 0.95,
            "reason": "Concrete CLI execution",
            "signals": [],
            "clarified_by_context": False,
        }

    has_macro_explicit = any(re.search(pat, q_lower) for pat in _EXPLICIT_DIRECTIONAL_PATTERNS)
    has_macro_framing = any(p in q_lower for p in _MACRO_FRAMING_PREFIXES)
    has_broad_noun = any(n in q_lower for n in _BROAD_SCOPE_NOUNS)
    has_target_files = bool(resolved_slots.get("target_files"))
    has_line_anchor = bool(re.search(r"(?:第\s*\d+\s*行|line\s+\d+)", q_lower))
    has_concrete_edit = any(c in q_lower for c in _CONCRETE_EDIT_CUES)

    # Pure read-only query without macro building/planning request
    if any(w in q_lower for w in _READONLY_QUERY_CUES) and not (has_macro_explicit or has_macro_framing):
        if not any(w in q_lower for w in ("删除", "重构", "修改", "写一个", "搭建", "构建", "做一个", "优化")):
            return {
                "is_directional": False,
                "confidence": 0.88,
                "reason": "Read-only query",
                "signals": [],
                "clarified_by_context": False,
            }

    # Concrete target-file edit/fix without macro redesign/planning
    if (has_target_files or has_line_anchor) and has_concrete_edit and not has_macro_explicit:
        return {
            "is_directional": False,
            "confidence": 0.90,
            "reason": "Concrete file/line modification",
            "signals": [],
            "clarified_by_context": False,
        }

    # Bounded single-unit algorithm/test request without broad system nouns
    if any(w in q_lower for w in ("快速排序", "冒泡排序", "二分查找", "quicksort", "binary search")) and not has_macro_explicit:
        return {
            "is_directional": False,
            "confidence": 0.90,
            "reason": "Bounded single-algorithm implementation",
            "signals": [],
            "clarified_by_context": False,
        }

    signals: List[str] = []
    if has_macro_explicit:
        signals.append("explicit_directional_cue")
    if has_macro_framing and has_broad_noun and not (has_target_files and has_line_anchor):
        signals.append("open_macro_goal")
    if ("方向" in q_lower or "规划" in q_lower) and has_broad_noun and not has_line_anchor:
        signals.append("directional_planning_noun")

    if signals:
        conf = 0.94 if "explicit_directional_cue" in signals else 0.91
        return {
            "is_directional": True,
            "confidence": conf,
            "reason": "Directional / macro instruction requires detail and planning alignment before execution",
            "signals": signals,
            "clarified_by_context": False,
        }

    return {
        "is_directional": False,
        "confidence": 0.85,
        "reason": "Concrete or non-directional instruction",
        "signals": [],
        "clarified_by_context": False,
    }


def is_readonly_query(instruction: str, context: str = "") -> bool:
    """Return True when instruction is a pure read-only / research query without mutation intent."""
    clean = clean_user_text(instruction)
    if not clean:
        return False
    q_lower = clean.lower()
    stripped = _NON_MUTATION_COMPOUND_RE.sub("", q_lower)
    if any(w in stripped for w in _MUTATION_ACTION_BLOCKERS):
        return False
    dir_eval = evaluate_directional_intent(clean, context=context)
    if dir_eval.get("is_directional"):
        return False
    return any(w in q_lower for w in _READONLY_QUERY_CUES)


def _is_ask_question_answer(step: Dict[str, Any]) -> bool:
    """Detect whether a transcript GENERIC step contains an ask_question user response."""
    if step.get("type") != "GENERIC" or step.get("status") != "DONE":
        return False
    content = str(step.get("content") or "")
    if content.startswith("Encountered error"):
        return False
    return bool(re.search(r"(?:^|\n)A1:", content))


def _step_has_mutating_tool(step: Dict[str, Any]) -> bool:
    for tc in step.get("tool_calls") or []:
        name = str(tc.get("name") or "").lower()
        if name in MUTATING_OR_EXEC_TOOLS:
            return True
    return False


def _step_has_ask_question(step: Dict[str, Any]) -> bool:
    for tc in step.get("tool_calls") or []:
        if str(tc.get("name") or "").lower() == "ask_question":
            return True
    return False


def inspect_transcript_directional_state(
    transcript_path: Optional[str],
    conversation_id: str = "",
    lock_store_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Inspect transcript.jsonl (and optional disk lock fallback) for directional lock state."""
    p = Path(transcript_path) if transcript_path else None
    if p and p.is_file():
        steps: List[Dict[str, Any]] = []
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        steps.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            steps = []

        user_indices = [
            i for i, st in enumerate(steps)
            if st.get("type") == "USER_INPUT" or st.get("source") == "USER_EXPLICIT"
        ]
        if user_indices:
            last_idx = user_indices[-1]
            last_query = clean_user_text(str(steps[last_idx].get("content") or ""))
            post_steps = steps[last_idx + 1:]

            # Check if ask_question was called AND answered after the latest USER_INPUT
            ask_seen = False
            ask_answered = False
            for st in post_steps:
                if _step_has_ask_question(st):
                    ask_seen = True
                elif ask_seen and _is_ask_question_answer(st):
                    ask_answered = True
                    break

            # Walk back over unresolved clarification follow-ups so missing UI / cut-off reports do not unlock gate
            prior_clarification_resolved = False
            inherited_directional_lock = False
            is_followup_query = _is_unresolved_clarification_followup(last_query)
            if len(user_indices) >= 2 and not ask_answered:
                anchor_idx: Optional[int] = None
                for k in range(len(user_indices) - 2, max(-1, len(user_indices) - 7), -1):
                    curr_u = user_indices[k]
                    next_u = user_indices[k + 1]
                    seg = steps[curr_u + 1 : next_u]
                    if any(_step_has_mutating_tool(st) or _is_ask_question_answer(st) for st in seg):
                        break
                    cand_q = clean_user_text(str(steps[curr_u].get("content") or ""))
                    if _is_unresolved_clarification_followup(cand_q):
                        continue
                    if evaluate_directional_intent(cand_q)["is_directional"]:
                        anchor_idx = curr_u
                    break

                if anchor_idx is not None:
                    between_anchor = steps[anchor_idx + 1 : last_idx]
                    had_planner_text = any(
                        st.get("type") == "PLANNER_RESPONSE" and str(st.get("content") or "").strip()
                        for st in between_anchor
                    )
                    had_mutation_or_ask = any(
                        _step_has_mutating_tool(st) or _is_ask_question_answer(st)
                        for st in between_anchor
                    )
                    if is_followup_query:
                        inherited_directional_lock = True
                    elif had_planner_text and not had_mutation_or_ask:
                        prior_clarification_resolved = True

            if inherited_directional_lock and not ask_answered:
                ctx = "[DIRECTIONAL_FOLLOWUP_LOCKED]"
            elif ask_answered or prior_clarification_resolved:
                ctx = "[CLARIFICATION_RESOLVED]"
            else:
                ctx = ""
            eval_res = evaluate_directional_intent(last_query, context=ctx)
            is_locked = bool(eval_res["is_directional"] and not ask_answered)

            return {
                "latest_user_query": last_query,
                "user_step_index": steps[last_idx].get("step_index", last_idx),
                "ask_question_answered": ask_answered,
                "prior_clarification_resolved": prior_clarification_resolved,
                "dialogue_context": ctx,
                "is_directional": bool(eval_res["is_directional"]),
                "is_locked": is_locked,
                "signals": eval_res["signals"],
                "reason": eval_res["reason"],
            }

    # Fallback: read persisted lock state by conversation_id when transcript is unavailable
    if conversation_id:
        lock_state = read_directional_lock(conversation_id, lock_store_path=lock_store_path)
        if lock_state and lock_state.get("locked"):
            return {
                "latest_user_query": lock_state.get("query", ""),
                "user_step_index": lock_state.get("user_step_index", -1),
                "ask_question_answered": False,
                "prior_clarification_resolved": False,
                "dialogue_context": "",
                "is_directional": True,
                "is_locked": True,
                "signals": lock_state.get("signals", ["persisted_lock"]),
                "reason": lock_state.get("reason", "Persisted directional lock active"),
            }

    return {
        "latest_user_query": "",
        "user_step_index": -1,
        "ask_question_answered": False,
        "prior_clarification_resolved": False,
        "dialogue_context": "",
        "is_directional": False,
        "is_locked": False,
        "signals": [],
        "reason": "No active directional lock",
    }


def write_directional_lock(
    conversation_id: str,
    locked: bool,
    query: str = "",
    user_step_index: int = -1,
    signals: Optional[List[str]] = None,
    reason: str = "",
    lock_store_path: Optional[Path] = None,
) -> None:
    """Persist per-conversation directional lock state across short-lived hook invocations."""
    if not conversation_id:
        return
    store = lock_store_path or DEFAULT_LOCK_STORE
    try:
        store.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {}
        if store.is_file():
            try:
                data = json.loads(store.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except Exception:
                data = {}
        if locked:
            data[conversation_id] = {
                "locked": True,
                "query": query[:500],
                "user_step_index": user_step_index,
                "signals": signals or [],
                "reason": reason,
                "updated_at": time.time(),
            }
        else:
            data.pop(conversation_id, None)
        store.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def read_directional_lock(
    conversation_id: str,
    lock_store_path: Optional[Path] = None,
    max_age_sec: float = 1800.0,
) -> Optional[Dict[str, Any]]:
    """Read persisted directional lock record for conversation_id, auto-expiring stale locks."""
    if not conversation_id:
        return None
    store = lock_store_path or DEFAULT_LOCK_STORE
    if not store.is_file():
        return None
    try:
        data = json.loads(store.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            rec = data.get(conversation_id)
            if isinstance(rec, dict):
                updated_at = float(rec.get("updated_at") or 0.0)
                if max_age_sec > 0 and updated_at > 0 and (time.time() - updated_at) > max_age_sec:
                    data.pop(conversation_id, None)
                    store.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    return None
                return rec
    except Exception:
        pass
    return None


def is_readonly_tool_call(tool_name: str, args: Optional[Dict[str, Any]] = None) -> bool:
    """Check whether a tool invocation is strictly read-only or user clarification."""
    name = (tool_name or "").lower().strip()
    if name in READONLY_TOOLS:
        return True
    if name == "call_mcp_tool":
        mcp_tool = str((args or {}).get("ToolName") or "").lower().strip()
        if mcp_tool in READONLY_MCP_TOOLS:
            return True
    return False


def evaluate_directional_tool_gate(
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    locked: bool = False,
) -> Optional[Dict[str, str]]:
    """Enforce PreToolUse deny barrier while directional planning lock is active."""
    if not locked:
        return None
    if is_readonly_tool_call(tool_name, args):
        return None
    name = (tool_name or "").lower().strip()
    if name in MUTATING_OR_EXEC_TOOLS or name == "call_mcp_tool":
        return {
            "decision": "deny",
            "reason": (
                "[JIT_CAPSULE: DIRECTIONAL_PLANNING_GATE] 当前指令被上下文 JEV 判定为方向性指令 "
                "(DIRECTIONAL_PLANNING)，在用户确认具体细节与规划要求前，物理门禁禁止调用写盘工具及 "
                "run_command。请先使用 read_file / vault_search / shelf_search 只读勘察现状，"
                "再以 <= 1500 字核心摘要概述边界并通过 ask_question（或简短文字）与用户对齐方案与验收标准，"
                "切勿同轮堆砌超长文本后再调 ask_question 以防输出截断。"
            ),
        }
    return None


def format_directional_hud_block(is_locked: bool, signals: Optional[List[str]] = None) -> str:
    """Format the zero-emoji Attention HUD block for directional pre-alignment."""
    if not is_locked:
        return ""
    sig_str = ", ".join(signals or ["directional_intent"])
    return "\n".join([
        "- [上下文 JEV 判定 - 方向性前置对齐] 状态: `DIRECTIONAL_PLANNING` | 门禁: `PRE_TOOL_USE_LOCKED`",
        f"  * 触发特征: {sig_str}",
        "  * [硬约束] 严禁马上开始写代码或执行命令 (write_file / edit_file / run_command 已被物理阻断)",
        "  * [协作流程] 1) 先用 read_file / vault_search / shelf_search 静默只读勘察现状; "
        "2) 若调用 ask_question 发起确认，同轮正文严禁超过 1500 字（优先短摘要或无正文直接调用 ask_question，防止高思考预算+长正文触发 STOP_PATTERN 丢失工具调用）; "
        "3) 获用户确认后再展开完整规划并实施",
    ])
