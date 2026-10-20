#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Epistemic Evidence Fusion & Weight Matrix Engine.

Provides context-aware 5-dimensional weight distribution across evidence channels:
- w_exec: Physical execution truth (compiler, tests, exit_code 0)
- w_official: Internet official authoritative docs (whitelisted primary domains)
- w_web: General internet/community discussions (forums, blogs)
- w_local: Local codebase, config files, memory vault
- w_model: Underlying decision model causal advisory
Includes strict non-linear conflict resolution and hard veto masks.
"""

from __future__ import annotations

import re
import urllib.parse
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class EpistemicContext(str, Enum):
    """Task environment taxonomy determining evidence credibility distribution."""
    OPEN_WORLD_FACT = "open_world_fact"          # Dynamic market releases, API specs, pricing, official news
    CLOSED_WORLD_CODE = "closed_world_code"      # Pure local code, algo, refactoring, tests, AST check
    ECOSYSTEM_DEBUG = "ecosystem_debug"          # Framework quirks, edge cases, community troubleshooting, stack traces
    SAFETY_GOVERNANCE = "safety_governance"      # High-risk mutation, danger interception, resource saturation


class AuthoritativeDomainEngine:
    """Whitelists and verifies whether an information source belongs to official primary authorities."""

    PRIMARY_VENDOR_DOMAINS = [
        "openai.com", "platform.openai.com",
        "anthropic.com", "docs.anthropic.com",
        "deepseek.com", "api-docs.deepseek.com",
        "ai.google.dev", "cloud.google.com",
        "aliyun.com", "modelscope.cn",
        "open.bigmodel.cn", "zhipuai.cn",
        "moonshot.cn", "platform.moonshot.cn",
        "baichuan-ai.com", "minimaxi.com",
        "meta.com", "llama.meta.com",
    ]

    STANDARDS_AND_RUNTIME_DOMAINS = [
        "python.org", "docs.python.org",
        "docker.com", "docs.docker.com",
        "pytorch.org", "sqlite.org",
        "fastapi.tiangolo.com", "docs.pydantic.dev",
        "ietf.org", "w3.org", "tools.ietf.org",
        "huggingface.co", "github.com",
    ]

    @classmethod
    def is_authoritative(cls, url_or_domain: str) -> Tuple[bool, str]:
        """Check if target domain qualifies as official authoritative primary source."""
        if not url_or_domain:
            return False, "unknown"
        clean = url_or_domain.lower().strip()
        if "://" in clean:
            try:
                parsed = urllib.parse.urlparse(clean)
                clean = parsed.netloc or parsed.path
            except Exception:
                pass
        clean = clean.split("/")[0].split(":")[0]

        for domain in cls.PRIMARY_VENDOR_DOMAINS:
            if clean == domain or clean.endswith(f".{domain}"):
                return True, "primary_vendor"

        for domain in cls.STANDARDS_AND_RUNTIME_DOMAINS:
            if clean == domain or clean.endswith(f".{domain}"):
                return True, "standards_and_runtime"

        return False, "third_party_or_community"


class EpistemicWeightMatrix:
    """Computes context-aware 5-dimensional weight distribution across evidence channels."""

    BASE_WEIGHTS: Dict[EpistemicContext, Dict[str, float]] = {
        EpistemicContext.OPEN_WORLD_FACT: {
            "w_exec": 0.00,
            "w_official": 0.70,
            "w_web": 0.15,
            "w_local": 0.05,
            "w_model": 0.10,
        },
        EpistemicContext.CLOSED_WORLD_CODE: {
            "w_exec": 0.40,
            "w_official": 0.05,
            "w_web": 0.00,
            "w_local": 0.40,
            "w_model": 0.15,
        },
        EpistemicContext.ECOSYSTEM_DEBUG: {
            "w_exec": 0.25,
            "w_official": 0.30,
            "w_web": 0.30,
            "w_local": 0.10,
            "w_model": 0.05,
        },
        EpistemicContext.SAFETY_GOVERNANCE: {
            "w_exec": 0.10,
            "w_official": 0.00,
            "w_web": 0.00,
            "w_local": 0.30,
            "w_model": 0.60,
        },
    }

    @classmethod
    def get_weights(cls, context: EpistemicContext) -> Dict[str, float]:
        """Retrieve base normalized weights for given task context."""
        return dict(cls.BASE_WEIGHTS.get(context, cls.BASE_WEIGHTS[EpistemicContext.CLOSED_WORLD_CODE]))


class EpistemicArbiter:
    """Multi-source evidence arbiter preventing blind belief in models, local cache, or unverified claims."""

    def __init__(self):
        self.domain_engine = AuthoritativeDomainEngine()

    def detect_context(self, instruction: str, context_snippet: str = "") -> EpistemicContext:
        """Deterministically detect task environment archetype from physical cues."""
        text = f"{instruction} {context_snippet}".lower()

        # 1. Safety Governance
        if any(w in text for w in ("删除", "rmdir", "delete", "rm -rf", "清空", "format", "覆盖配置", "高危", "危险", "写盘", "破坏", "oom", "drop database")):
            return EpistemicContext.SAFETY_GOVERNANCE

        # 2. Open World Fact (Latest models, pricing, official docs, release specs)
        if any(w in text for w in ("最新模型", "官方文档", "发布", "在售", "定价", "费率", "pricing", "一手语料", "release", "spec", "官网", "各厂商")):
            return EpistemicContext.OPEN_WORLD_FACT

        # 3. Ecosystem Debugging (Exceptions, stack traces, quirks)
        if any(w in text for w in ("winerror", "attributeerror", "traceback", "stacktrace", "踩坑", "排错", "报错", "issue", "crash")):
            return EpistemicContext.ECOSYSTEM_DEBUG

        # 4. Default: Closed World Code (Algorithms, refactoring, unit tests, local functions)
        return EpistemicContext.CLOSED_WORLD_CODE

    def arbitrate(
        self,
        instruction: str,
        context_snippet: str = "",
        model_recommendation: Optional[Dict[str, Any]] = None,
        local_evidence: Optional[Dict[str, Any]] = None,
        live_evidence: Optional[Dict[str, Any]] = None,
        execution_telemetry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute non-linear conflict resolution and return dynamic weight assignment with advice."""
        ctx = self.detect_context(instruction, context_snippet)
        weights = EpistemicWeightMatrix.get_weights(ctx)
        masks_applied: List[str] = []

        # Rule 1: Temporal Invalidation Mask
        if ctx == EpistemicContext.OPEN_WORLD_FACT:
            if live_evidence and live_evidence.get("has_official_source"):
                original_local = weights["w_local"]
                weights["w_local"] = 0.0
                weights["w_official"] = round(weights["w_official"] + original_local, 2)
                masks_applied.append(f"TEMPORAL_INVALIDATION: Stale local cache masked from {original_local} to 0.0")

        # Rule 2: Empirical Reality Veto (Physical Execution Dominance)
        if execution_telemetry:
            exit_code = execution_telemetry.get("exit_code") if execution_telemetry.get("exit_code") is not None else execution_telemetry.get("exitCode")
            if exit_code is not None and exit_code != 0:
                weights["w_exec"] = 0.80
                weights["w_local"] = min(weights["w_local"], 0.10)
                weights["w_model"] = min(weights["w_model"], 0.05)
                weights["w_web"] = min(weights["w_web"], 0.05)
                masks_applied.append(f"EMPIRICAL_VETO: Physical exit_code={exit_code} != 0. Execution error overrides all verbal claims.")

        # Rule 3: Safety Breach Veto
        if model_recommendation and model_recommendation.get("danger_level", 1) >= 4:
            weights["w_model"] = 0.85
            weights["w_local"] = min(weights["w_local"], 0.10)
            weights["w_web"] = 0.0
            weights["w_official"] = 0.0
            weights["w_exec"] = min(weights["w_exec"], 0.05)
            masks_applied.append(f"SAFETY_VETO: High operational hazard detected (level {model_recommendation.get('danger_level')}). Interception priority active.")

        # Re-normalize weights to sum to 1.0
        tot = sum(weights.values())
        if tot > 0:
            weights = {k: round(v / tot, 3) for k, v in weights.items()}

        primary_channel = max(weights.items(), key=lambda x: x[1])[0]
        channel_desc = {
            "w_exec": "物理执行事实 (测试退出码 / 编译器 AST)",
            "w_official": "互联网一手官方权威文档 (白名单主域)",
            "w_web": "一般互联网社区讨论与开发者论坛",
            "w_local": "本地源码 / 配置文件 / 知识库",
            "w_model": "决策模型前置因果判断与安全哨兵",
        }

        guidance = (
            f"当前任务环境识别为 [{ctx.value.upper()}]。最高置信渠道为 [{channel_desc.get(primary_channel)}] "
            f"(权重: {weights.get(primary_channel, 0.0):.1%})。严禁主模型单一信赖决策模型先验或本地陈旧缓存。"
        )

        return {
            "status": "success",
            "context": ctx.value,
            "weights": weights,
            "primary_authority": primary_channel,
            "primary_authority_desc": channel_desc.get(primary_channel),
            "masks_applied": masks_applied,
            "guidance": guidance,
        }
