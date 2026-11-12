# -*- coding: utf-8 -*-
"""DependencyRouterDecider: 本地依赖适用路由指引与架构推荐引擎.

基于本地静态映射矩阵 (data/dependency_routing_matrix.json) 与 Jev 因果仲裁能力，
识别模型开发需求中的领域场景与目标环境，动态输出推荐生产依赖与反模式规约。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base import JevDeciderBase

MATRIX_PATH = Path(__file__).resolve().parent.parent / "data" / "dependency_routing_matrix.json"


class DependencyRouterDecider:
    def __init__(self, base: Optional[JevDeciderBase] = None, matrix_path: Optional[Path] = None):
        self.base = base
        self.matrix_path = matrix_path or MATRIX_PATH
        self._matrix_cache: Optional[Dict[str, Any]] = None

    def _load_matrix(self) -> Dict[str, Any]:
        if self._matrix_cache is not None:
            return self._matrix_cache
        if not self.matrix_path.exists():
            return {"environments": {}, "routes": []}
        try:
            with open(self.matrix_path, "r", encoding="utf-8") as f:
                self._matrix_cache = json.load(f)
                return self._matrix_cache
        except Exception:
            return {"environments": {}, "routes": []}

    def detect_ecosystem(self, prompt: str, context: str = "") -> Optional[str]:
        text = f"{prompt} {context}".lower()
        has_py = any(k in text for k in ("python", "fastapi", "django", "flask", "pydantic", "sqlalchemy", "celery", ".py"))
        has_js = any(k in text for k in ("javascript", "typescript", "node", "react", "vue", "fastify", "nestjs", "pnpm", "npm", ".ts", ".tsx", ".js"))
        has_go = any(k in text for k in ("golang", "go ", "gin", "fiber", "gorm", "sqlc", ".go"))

        if has_py and not (has_js or has_go):
            return "python"
        if has_js and not (has_py or has_go):
            return "javascript"
        if has_go and not (has_py or has_js):
            return "go"
        return None

    def match_guidance(
        self,
        prompt: str,
        context: str = "",
        ecosystem_hint: Optional[str] = None,
        max_matches: int = 2
    ) -> List[Dict[str, Any]]:
        """依据用户指令与上下文匹配推荐的依赖适用路由指引."""
        matrix_data = self._load_matrix()
        routes = matrix_data.get("routes", [])
        environments = matrix_data.get("environments", {})

        p_lower = prompt.lower()
        c_lower = context.lower()
        combined = f"{p_lower} {c_lower}"

        detected_eco = ecosystem_hint or self.detect_ecosystem(prompt, context)
        results = []

        for route in routes:
            trigger_keywords = route.get("trigger_keywords", [])
            hit = any(kw in combined for kw in trigger_keywords)
            if not hit:
                continue

            matrix = route.get("matrix", {})
            # 确定匹配的生态分支
            target_eco = detected_eco
            if target_eco and target_eco in matrix:
                eco_entry = matrix[target_eco]
            elif "react" in combined and "react" in matrix:
                target_eco = "react"
                eco_entry = matrix["react"]
            elif "vue" in combined and "vue" in matrix:
                target_eco = "vue"
                eco_entry = matrix["vue"]
            elif "python" in matrix:
                target_eco = "python"
                eco_entry = matrix["python"]
            else:
                target_eco = next(iter(matrix.keys()))
                eco_entry = matrix[target_eco]

            env_key = eco_entry.get("environment", "")
            env_info = environments.get(env_key, {})

            results.append({
                "domain": route.get("domain", ""),
                "category": route.get("category", ""),
                "ecosystem": target_eco,
                "tier": eco_entry.get("tier", "Tier 1"),
                "recommended": eco_entry.get("recommended", []),
                "environment_name": env_info.get("name", env_key),
                "runtime": env_info.get("runtime", "默认"),
                "isolation": env_info.get("isolation", "项目级隔离"),
                "rationale": eco_entry.get("rationale", ""),
                "anti_patterns": eco_entry.get("anti_patterns", []),
            })

            if len(results) >= max_matches:
                break

        return results

    def format_hud_guidance(self, guidances: List[Dict[str, Any]]) -> str:
        """生成 Attention HUD 适用的依赖指引注入 Markdown 文本 (无任何 Emoji)."""
        if not guidances:
            return ""
        lines = []
        for g in guidances:
            cat = g.get("category", g.get("domain", "开发需求"))
            rec = ", ".join(g.get("recommended", []))
            env = g.get("runtime", g.get("environment_name", "标准环境"))
            rat = g.get("rationale", "")
            lines.append(f"- [依赖路由指引] {cat}: 推荐 `{rec}` | 环境: `{env}` | 规范: {rat}")
        return "\n".join(lines)

    def route_dependencies(
        self,
        prompt: str,
        context: str = "",
        ecosystem: Optional[str] = None
    ) -> Dict[str, Any]:
        """门面调用方法，供 MCP 工具或控制器直接消费."""
        guidances = self.match_guidance(prompt, context=context, ecosystem_hint=ecosystem)
        hud_text = self.format_hud_guidance(guidances)
        return {
            "matched_count": len(guidances),
            "guidances": guidances,
            "hud_text": hud_text,
        }
