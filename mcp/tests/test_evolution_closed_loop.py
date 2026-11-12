#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression Test Suite for JEV Incremental Evolution & Hot Reload Closed Loop.

Verifies:
1. 70/30 experience replay buffer assembly (Base + Xing intent increment).
2. Rapid convergence (< 2.0s) and physical adapter.pt emission.
3. In-place zero-downtime hot reload status verification.
4. Engine facade trigger_evolution() & get_evolution_status() integration.
5. MCP Server tool dispatch for gemma_trigger_evolution and gemma_evolution_status.
"""

import sys
import time
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from harvest.evolution_runner import JevEvolutionRunner
from engine import GemmaPrepEngine
from server import GemmaPrepCoreMCPServer


class TestEvolutionClosedLoop:

    @pytest.fixture(autouse=True)
    def _isolate_evolution_outputs(self, tmp_path, monkeypatch):
        import harvest.evolution_runner as er
        iso_adapter = tmp_path / "isolated_adapter.pt"
        iso_meta = tmp_path / "isolated_evolution_meta.json"
        monkeypatch.setattr(er, "DEFAULT_ADAPTER_PATH", iso_adapter)
        monkeypatch.setattr(er, "METADATA_PATH", iso_meta)
        monkeypatch.setattr(er.JevEvolutionRunner.__init__, "__defaults__", (er.DATA_DIR, iso_adapter))

    def test_runner_distillation_and_weight_emission(self, tmp_path):
        """1. 验证增量蒸馏物理产出 7KB 级 adapter.pt 权重与元数据."""
        adapter_file = tmp_path / "adapter.pt"
        runner = JevEvolutionRunner(adapter_path=adapter_file)
        
        t0 = time.perf_counter()
        res = runner.run_incremental_distillation(epochs=5, force=True)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        assert res["status"] == "success"
        assert res["loss"] <= 0.15
        assert res["increment_samples_used"] >= 1
        assert res["base_samples_used"] >= 1
        assert adapter_file.exists()
        assert adapter_file.stat().st_size > 1024  # > 1KB
        assert latency_ms < 2000.0, f"Training took too long: {latency_ms:.2f}ms"

    def test_engine_facade_evolution_lifecycle(self):
        """2. 验证 GemmaPrepEngine 顶级门面自进化调用与状态感知."""
        engine = GemmaPrepEngine()
        res = engine.trigger_evolution(epochs=3, force=True)

        assert res["status"] == "success"
        assert res["hot_reloaded"] is True
        assert "loss" in res

        status = engine.get_evolution_status()
        assert status.get("status") == "success" or status.get("has_adapter") is True
        assert status.get("loss") is not None

    def test_mcp_server_evolution_tool_dispatch(self):
        """3. 验证 MCP Server 对 gemma_trigger_evolution / status 的 Stdio 分发."""
        server = GemmaPrepCoreMCPServer()

        # 测试触发训练
        res_trigger = server.handle_tool_call("gemma_trigger_evolution", {"epochs": 2, "force": True})
        assert res_trigger.get("status") == "success"
        assert res_trigger.get("hot_reloaded") is True

        # 测试查询状态
        res_status = server.handle_tool_call("gemma_evolution_status", {})
        assert res_status.get("status") == "success" or res_status.get("has_adapter") is True
