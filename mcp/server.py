#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GemmaPrepCoreMCPServer:
标准 MCP (Model Context Protocol) 服务，向 Gemini / Antigravity Agent 提供：
1. gemma_intent_classify: 指令前置意图分类
2. gemma_memory_recall: 跨源数据与历史会话记忆混合召回
3. gemma_jev_decide: Jev 三原语 (noul/choice/score) 因果决策判定
4. gemma_assemble_context: 上下文装配流水线 (Attention HUD 组装)
5. gemma_hardware_status: GPU 显存与硬件安全感知
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# 确保导入当前目录模块
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine import GemmaPrepEngine
from pipeline import ContextAssemblyPipeline


class GemmaPrepCoreMCPServer:
    def __init__(self):
        self.engine = GemmaPrepEngine()
        self.pipeline = ContextAssemblyPipeline(self.engine)

    def get_tools_manifest(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "gemma_intent_classify",
                "description": "Pre-instruction intent classifier powered by EmbeddingGemma-Jev. Categorizes user goals into CODE_MUTATE, READONLY_QUERY, SYSTEM_COMMAND, MEDIA_COMFYUI, RAG_RECALL, or DANGEROUS_WRITE.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "The user's prompt or target instruction to evaluate."
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional surrounding context or file state (up to 8K tokens)."
                        }
                    },
                    "required": ["instruction"]
                }
            },
            {
                "name": "gemma_memory_recall",
                "description": "Hybrid recall across 2,095 physical dialogue history chunks, memory-vault long-term knowledge cards, and local skills library.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search keywords or user intent query."
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top results to return (default 4)."
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "gemma_jev_decide",
                "description": "Deterministic Jev causal decision judge: 'noul' (binary gate/anomaly check), 'choice' (action routing), or 'score' (1-5 risk grading).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["noul", "choice", "score"],
                            "description": "Decision primitive mode."
                        },
                        "question": {
                            "type": "string",
                            "description": "Statement or question to judge."
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of options for 'choice' or 'score' mode (e.g. [{'t': 'Option A', 'd': 'Desc'}])."
                        },
                        "state": {
                            "type": "string",
                            "description": "Detailed background context (error trace, code snippet, hardware state)."
                        }
                    },
                    "required": ["mode", "question"]
                }
            },
            {
                "name": "gemma_assemble_context",
                "description": "End-to-end Context Assembly Pipeline. Combines intent parsing, memory recall, and GPU hardware status into a ready-to-inject Attention HUD card.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "User instruction or prompt."
                        },
                        "state": {
                            "type": "string",
                            "description": "Optional environment or task state."
                        },
                        "include_hardware": {
                            "type": "boolean",
                            "description": "Whether to query GPU VRAM status (default true)."
                        }
                    },
                    "required": ["instruction"]
                }
            },
            {
                "name": "gemma_hardware_status",
                "description": "Query live NVIDIA GPU VRAM occupancy, ComfyUI concurrency status, and hardware danger level.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "gemma_post_task_audit",
                "description": "Post-task codebase audit and reorganization judge powered by EmbeddingGemma-2 & Jev. Audits line counts against the 800-line limit, detects AST structural changes, decides PASS / SYNC_VAULT / WARN_REFACTOR, and formats memory-vault architecture payloads.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "workspace_path": {
                            "type": "string",
                            "description": "Absolute path to the workspace root directory."
                        },
                        "changed_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of changed file paths in this session."
                        },
                        "exit_code": {
                            "type": "integer",
                            "description": "Exit code of physical tests/build (default 0)."
                        },
                        "session_goal": {
                            "type": "string",
                            "description": "Optional original goal or instruction of the task."
                        },
                        "max_line_limit": {
                            "type": "integer",
                            "description": "Max file line budget threshold (default 800)."
                        }
                    },
                    "required": ["workspace_path"]
                }
            },
            {
                "name": "gemma_command_audit",
                "description": "Static semantic safety auditor for shell/terminal commands powered by Gemma-Jev. Evaluates destructive danger levels (1-5) and flags dangerous deletions, process kills, or system mutations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command_line": {
                            "type": "string",
                            "description": "Exact command line string to evaluate."
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Optional working directory context."
                        },
                        "is_daemon": {
                            "type": "boolean",
                            "description": "Whether the command is intended to run as a daemon process."
                        }
                    },
                    "required": ["command_line"]
                }
            },
            {
                "name": "gemma_failure_triage",
                "description": "Millisecond root-cause classifier for test failures, crash tracebacks, and non-zero exit codes. Outputs failure category and targeted remediation capsules.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "traceback_text": {
                            "type": "string",
                            "description": "Full or partial error traceback / stderr string."
                        },
                        "exit_code": {
                            "type": "integer",
                            "description": "Process non-zero exit code (default 1)."
                        }
                    },
                    "required": ["traceback_text"]
                }
            },
            {
                "name": "gemma_memory_gate",
                "description": "High-entropy novelty filter and gatekeeper for memory-vault ingestion. Prevents low-value chatter and noise from polluting the long-term knowledge base.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Title of the proposed memory node."
                        },
                        "content": {
                            "type": "string",
                            "description": "Main body text of the memory node."
                        },
                        "track": {
                            "type": "string",
                            "description": "Target track (architecture, errors, trajectory, etc.)."
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of tags associated with the memory."
                        }
                    },
                    "required": ["title", "content"]
                }
            },
            {
                "name": "gemma_model_route",
                "description": "Cognitive complexity analyzer and model tier router. Recommends the optimal model tier (LOCAL_LIGHT, CLOUD_FLASH, CLOUD_PRO) for maximum cost-performance efficiency.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Target prompt or user task instruction."
                        },
                        "context_len": {
                            "type": "integer",
                            "description": "Length of context in characters/tokens."
                        }
                    },
                    "required": ["prompt"]
                }
            }
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "gemma_intent_classify":
            return self.engine.classify_intent(args.get("instruction", ""), args.get("context", ""))
        elif name == "gemma_memory_recall":
            top_k = args.get("top_k", 4)
            return {"results": self.engine.hybrid_recall(args.get("query", ""), top_k=top_k)}
        elif name == "gemma_jev_decide":
            return self.engine.jev_decide(
                mode=args.get("mode", "noul"),
                question=args.get("question", ""),
                options=args.get("options", []),
                state=args.get("state", "")
            )
        elif name == "gemma_assemble_context":
            return self.pipeline.assemble(
                instruction=args.get("instruction", ""),
                state=args.get("state", ""),
                include_recall=True,
                include_hardware=args.get("include_hardware", True)
            )
        elif name == "gemma_hardware_status":
            return self.engine.get_hardware_status()
        elif name == "gemma_post_task_audit":
            return self.engine.post_task_codebase_audit(
                workspace_path=args.get("workspace_path", ""),
                changed_files=args.get("changed_files"),
                exit_code=args.get("exit_code", 0),
                session_goal=args.get("session_goal", ""),
                max_line_limit=args.get("max_line_limit", 800)
            )
        elif name == "gemma_command_audit":
            return self.engine.audit_command_safety(
                command_line=args.get("command_line", ""),
                cwd=args.get("cwd", ""),
                is_daemon=args.get("is_daemon", False)
            )
        elif name == "gemma_failure_triage":
            return self.engine.triage_failure(
                traceback_text=args.get("traceback_text", ""),
                exit_code=args.get("exit_code", 1)
            )
        elif name == "gemma_memory_gate":
            return self.engine.audit_memory_novelty(
                title=args.get("title", ""),
                content=args.get("content", ""),
                track=args.get("track", "all"),
                tags=args.get("tags")
            )
        elif name == "gemma_model_route":
            return self.engine.route_model_tier(
                prompt=args.get("prompt", ""),
                context_len=args.get("context_len", 0)
            )
        else:
            return {"error": f"Unknown tool: {name}"}

    def run(self):
        """MCP stdio JSON-RPC 主循环。"""
        stdin = sys.stdin.buffer
        stdout = sys.stdout.buffer

        while True:
            line = stdin.readline()
            if not line:
                break
            try:
                try:
                    line_str = line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    line_str = line.decode("utf-16").strip()
                if not line_str:
                    continue
                req = json.loads(line_str)
                msg_id = req.get("id")
                method = req.get("method")
                params = req.get("params", {})

                if method == "initialize":
                    res = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {
                                "name": "gemma-prep-core",
                                "version": "1.0.0"
                            }
                        }
                    }
                    stdout.write((json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
                    stdout.flush()

                elif method == "notifications/initialized":
                    pass

                elif method == "tools/list":
                    res = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "tools": self.get_tools_manifest()
                        }
                    }
                    stdout.write((json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
                    stdout.flush()

                elif method == "tools/call":
                    tool_name = params.get("name")
                    arguments = params.get("arguments", {})
                    out = self.handle_tool_call(tool_name, arguments)
                    res = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(out, ensure_ascii=False, indent=2)
                                }
                            ]
                        }
                    }
                    stdout.write((json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
                    stdout.flush()

                elif method == "ping":
                    res = {"jsonrpc": "2.0", "id": msg_id, "result": {}}
                    stdout.write((json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
                    stdout.flush()

                else:
                    if msg_id is not None:
                        res = {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "error": {"code": -32601, "message": f"Method not found: {method}"}
                        }
                        stdout.write((json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
                        stdout.flush()
            except Exception as e:
                pass


if __name__ == "__main__":
    server = GemmaPrepCoreMCPServer()
    server.run()
