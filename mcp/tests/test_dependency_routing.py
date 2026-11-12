# -*- coding: utf-8 -*-
"""Test suite for Local Dependency Routing Matrix and JEV Decision Guidance."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
CORE_DIR = HERE.parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from deciders.dependency_router import DependencyRouterDecider
from decision_substrate import DecisionSubstrate
from pipeline import ContextAssemblyPipeline
from engine import GemmaPrepEngine


def test_matrix_loading():
    router = DependencyRouterDecider()
    matrix = router._load_matrix()
    assert "routes" in matrix
    assert "environments" in matrix
    assert len(matrix["routes"]) >= 5
    assert "python_web_asgi" in matrix["environments"]
    assert "node_enterprise_web" in matrix["environments"]


def test_ecosystem_detection():
    router = DependencyRouterDecider()
    assert router.detect_ecosystem("写一个 FastAPI 接口") == "python"
    assert router.detect_ecosystem("开发一个 React 组件与状态管理") == "javascript"
    assert router.detect_ecosystem("用 Gin 框架构建低延迟微服务") == "go"


def test_dependency_routing_match_python_api():
    router = DependencyRouterDecider()
    guidances = router.match_guidance("开发一个高并发 API 服务，需要对入参进行 schema 校验")
    assert len(guidances) > 0
    domains = [g["domain"] for g in guidances]
    assert "web_api_service" in domains or "data_contract_validation" in domains

    first = guidances[0]
    assert "fastapi" in first["recommended"] or "pydantic" in first["recommended"]
    assert "tier" in first
    assert "anti_patterns" in first
    assert len(first["anti_patterns"]) > 0


def test_dependency_routing_match_frontend():
    router = DependencyRouterDecider()
    guidances = router.match_guidance("构建 React 前端界面，需要管理客户端全局状态与组件样式")
    assert len(guidances) > 0
    frontend_entry = next((g for g in guidances if g["domain"] == "frontend_ui_state"), None)
    assert frontend_entry is not None
    assert "zustand" in frontend_entry["recommended"]
    assert "tailwindcss" in frontend_entry["recommended"]


def test_hud_formatting_no_emojis():
    router = DependencyRouterDecider()
    guidances = router.match_guidance("开发 FastAPI 接口与数据库 ORM 持久化")
    hud_text = router.format_hud_guidance(guidances)
    assert "[依赖路由指引]" in hud_text
    # 严格红线断言: 绝对无任何 Emoji 符号
    for char in hud_text:
        code = ord(char)
        assert not (0x1F300 <= code <= 0x1FAFF or 0x2600 <= code <= 0x27BF), f"Found emoji code: {code}"


def test_decision_substrate_on_pre_turn_injection():
    substrate = DecisionSubstrate.get_instance()
    prompt = "请实现一个 Python Web API 模块，需要连接数据库并做 DTO 校验"
    grounding = substrate.on_pre_turn(session_id="test-session", prompt=prompt)

    assert grounding is not None
    assert len(grounding.dependency_guidance) > 0
    assert "[依赖路由指引]" in grounding.hud_markdown
    assert "fastapi" in grounding.hud_markdown.lower() or "pydantic" in grounding.hud_markdown.lower() or "sqlalchemy" in grounding.hud_markdown.lower()


def test_context_assembly_pipeline_injection():
    pipeline = ContextAssemblyPipeline()
    res = pipeline.assemble(instruction="构建异步任务队列与定时 worker 处理")
    assert "dependency_guidance" in res
    assert len(res["dependency_guidance"]) > 0
    assert "[依赖路由指引]" in res["hud_markdown"]
    assert "celery" in res["hud_markdown"].lower() or "bullmq" in res["hud_markdown"].lower() or "redis" in res["hud_markdown"].lower()
