# -*- coding: utf-8 -*-
"""CodebaseAuditDecider: 任务后代码仓库整理需求审查与 800 行硬上限审计."""
from __future__ import annotations

import os
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional
from .base import JevDeciderBase


class CodebaseAuditDecider:
    def __init__(self, base: JevDeciderBase):
        self.base = base

    def post_task_codebase_audit(
        self,
        workspace_path: str,
        changed_files: Optional[List[str]] = None,
        exit_code: int = 0,
        session_goal: str = "",
        max_line_limit: int = 800
    ) -> Dict[str, Any]:
        """
        依靠物理事实扫描 + EmbeddingGemma-Jev 因果判定，执行任务后代码仓库整理需求审查。
        """
        ws = Path(workspace_path).resolve()
        if not ws.exists() or not ws.is_dir():
            return {
                "error": f"Workspace directory not found: {workspace_path}",
                "audit_result": "PASS",
                "violations": [],
                "warnings": []
            }

        # 1. 搜集待审计文件
        target_files: List[Path] = []
        if changed_files:
            for f in changed_files:
                p = Path(f) if Path(f).is_absolute() else ws / f
                if p.is_file():
                    target_files.append(p.resolve())
        else:
            ignore_dirs = {".git", "__pycache__", ".nexus", "node_modules", "venv", ".venv", "dist", "build"}
            code_exts = {".py", ".ts", ".js", ".tsx", ".jsx", ".rs", ".go", ".cpp", ".c", ".h"}
            for root, dirs, files in os.walk(ws):
                dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
                for file in files:
                    ext = Path(file).suffix.lower()
                    if ext in code_exts:
                        target_files.append((Path(root) / file).resolve())

        # 2. 统计物理行数与 AST 符号
        file_metrics = []
        violations = []
        warnings = []
        total_lines = 0

        for file_path in target_files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
                line_count = len(text.splitlines())
                total_lines += line_count
                rel_path = file_path.relative_to(ws).as_posix()

                symbols = []
                if file_path.suffix == ".py":
                    try:
                        tree = ast.parse(text, filename=str(file_path))
                        for node in tree.body:
                            if isinstance(node, ast.ClassDef):
                                symbols.append({"name": node.name, "type": "class", "line": node.lineno})
                            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                symbols.append({"name": node.name, "type": "function", "line": node.lineno})
                    except Exception:
                        pass

                is_over = line_count > max_line_limit
                is_warn = (not is_over) and (line_count > int(max_line_limit * 0.85))

                item = {
                    "path": rel_path,
                    "lines": line_count,
                    "budget": max_line_limit,
                    "symbols": symbols,
                    "is_over_limit": is_over,
                    "is_warning": is_warn
                }
                file_metrics.append(item)

                if is_over:
                    violations.append({
                        "path": rel_path,
                        "lines": line_count,
                        "excess": line_count - max_line_limit
                    })
                elif is_warn:
                    warnings.append({
                        "path": rel_path,
                        "lines": line_count,
                        "margin": max_line_limit - line_count
                    })
            except Exception:
                pass

        file_metrics.sort(key=lambda x: x["lines"], reverse=True)

        # 3. 构造决策上下文，调用 Gemma-Jev 进行因果裁决
        options = [
            {"t": "PASS", "d": "轻量修改或只读任务，各文件严格在800行限额内，无需更新架构库。"},
            {"t": "SYNC_VAULT", "d": "任务物理验证通过(Exit Code 0)，模块或符号结构发生变更，建议将最新结构树与行数映射同步至 memory-vault (track='architecture')。"},
            {"t": "WARN_REFACTOR", "d": "检测到单文件突破800行硬约束或存在严重模块膨胀，必须唤醒 python-modularizer 执行模块化解耦重构。"}
        ]
        question = "请判定当前任务完成后代码仓库是否需要执行架构同步或重构整理："
        state = (
            f"目标: {session_goal}\n"
            f"退出码: {exit_code}\n"
            f"总文件数: {len(file_metrics)}, 总行数: {total_lines}\n"
            f"超800行违规数: {len(violations)}, 接近超限预警数: {len(warnings)}"
        )
        dec = self.base.jev_decide("choice", question, options, state)
        chosen_action = options[dec.get("pred", 0)]["t"]
        conf = dec.get("conf", 0.0)

        # 物理真理硬约束兜底校准:
        if violations:
            chosen_action = "WARN_REFACTOR"
            conf = max(conf, 0.95)
        elif exit_code == 0 and (
            len(file_metrics) >= 2 or any(k in session_goal.lower() for k in ("重构", "拆分", "解耦", "refactor", "decouple", "模块", "feature", "new"))
        ):
            if chosen_action == "PASS" and any(k in session_goal.lower() for k in ("重构", "拆分", "解耦", "refactor", "decouple", "结构")):
                chosen_action = "SYNC_VAULT"
                conf = max(conf, 0.88)

        # 4. 生成模型友好的结构树与符号映射矩阵 Markdown
        proj_name = ws.name
        tree_lines = [f"{proj_name}/"]
        for fm in file_metrics:
            flag = " [OVER 800!]" if fm["is_over_limit"] else (" [WARN]" if fm["is_warning"] else "")
            tree_lines.append(f"├── {fm['path']} ({fm['lines']} 行){flag}")
        tree_md = "\n".join(tree_lines)

        matrix_rows = [
            "| 物理文件 | 行数 | 状态 | 核心导出类与函数 |",
            "| :--- | :---: | :---: | :--- |"
        ]
        for fm in file_metrics:
            sym_str = ", ".join([f"`{s['name']}` ({s['type']})" for s in fm["symbols"][:5]]) or "-"
            status_tag = "超标 (>800行)" if fm["is_over_limit"] else ("预警 (>680行)" if fm["is_warning"] else "合规")
            matrix_rows.append(f"| `{fm['path']}` | {fm['lines']} | {status_tag} | {sym_str} |")
        matrix_md = "\n".join(matrix_rows)

        # 5. 准备 memory-vault 入库载荷 (若需要同步)
        vault_payload = {
            "node_id": f"arch:{proj_name.lower()}_topology",
            "track": "architecture",
            "stream": "code",
            "title": f"[架构] {proj_name} 模块化结构树与符号映射矩阵",
            "tags": ["architecture", proj_name.lower(), "code_tree", "symbol_matrix"],
            "content": f"### 1. 代码结构树\n```text\n{tree_md}\n```\n\n### 2. 符号物理映射矩阵\n{matrix_md}",
            "metadata": {
                "project_root": str(ws),
                "total_files": len(file_metrics),
                "total_lines": total_lines,
                "max_file_lines": max_line_limit,
                "violations_count": len(violations)
            }
        }

        return {
            "audit_result": chosen_action,
            "confidence": conf,
            "latency_ms": dec.get("ms", 0.0),
            "device": dec.get("device", "unknown"),
            "violations": violations,
            "warnings": warnings,
            "total_files": len(file_metrics),
            "total_lines": total_lines,
            "structure_tree": tree_md,
            "symbol_matrix": matrix_md,
            "file_metrics": file_metrics,
            "memory_vault_payload": vault_payload,
            "recommendation": (
                "文件行数超限，请立即调用 python-modularizer 进行 Shim 拆分治理。" if chosen_action == "WARN_REFACTOR"
                else ("建议调用 vault_store 将 memory_vault_payload 同步至知识库 architecture 专区。" if chosen_action == "SYNC_VAULT"
                      else "代码仓库合规且无宏观拓扑变动，无需同步。")
            )
        }
