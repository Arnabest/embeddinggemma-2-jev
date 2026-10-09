# Gemma-Prep-Core: Edge Decision Micro-Kernel

High-performance, edge-resident decision micro-kernel powered by **EmbeddingGemma-2** and an ultra-lightweight **7KB Jev Causal Decision Adapter**. Designed for AI agent hosts (such as Google Antigravity / Gemini agents), developer tooling, and automated workflows.

---

## Key Highlights

- **Sub-50ms Edge Latency**: Operates directly on local GPU (e.g., NVIDIA RTX 4060 Ti) with a tiny ~1.2GB VRAM footprint.
- **Zero Cloud Token Waste**: Offloads micro-decisions (command security, syntax failure triage, memory filtering, post-task audits) from expensive cloud LLMs.
- **Strict 800-Line Architecture**: Every component is modularized and adheres to strict line budget constraints (no monolithic files).
- **Physical Ground Truth**: Uses compiler gates, AST analysis, and deterministic fallbacks rather than ungrounded subjective self-evaluation.
- **Model Context Protocol (MCP)**: Native stdio JSON-RPC server providing 10 standard MCP tools.

---

## Architecture Topology

```text
gemma-prep-core/
├── engine.py                   # High-level Facade / Shim (100% backward compatible)
├── server.py                   # Standard MCP Stdio JSON-RPC Server
├── pipeline.py                 # Context Assembly Pipeline (Attention HUD)
├── hook_adapter.py             # Agent Loop Lifecycle Hooks (PreInvocation / PreToolUse)
│
├── deciders/                   # Specialized Causal Decision Micro-Kernels
│   ├── base.py                 # Jev Primitives (noul / choice / score) & Hardware Sensing
│   ├── command_guard.py        # Static Command Safety & Destructive Risk Grading (1-5)
│   ├── failure_triage.py       # Millisecond Root-Cause Traceback Classifier
│   ├── memory_gate.py          # Dual-Stream Memory Hygiene Gate (Code vs Interaction)
│   ├── codebase_audit.py       # Post-Task Reorganization & 800-Line Limit Auditor
│   └── model_router.py         # Cognitive Complexity Analyzer & Model Tier Router
│
└── recall/                     # Multi-Source Prior Recall Hub
    └── memory_hub.py           # Dialogue History, Skill Shelf, and Knowledge Vault Recall
```

---

## Decision Capabilities

### 1. Static Command Safety Guard (`gemma_command_audit`)
Evaluates shell commands before execution for destructive impact:
- Destructive deletes (`rm -rf`, `Remove-Item -Recurse -Force`, `drop table`)
- Process kills (`kill -9`, `taskkill /f`)
- System mutations (`reg add`, `net user`, `bcdedit`)
- Assigns 1-5 danger levels and triggers interactive confirmation when $\ge 4$.

### 2. Traceback Triage & Failure Classifier (`gemma_failure_triage`)
Parses non-zero exit code tracebacks within milliseconds into actionable categories:
- `DEPENDENCY_MISSING`: Missing Python packages or DLLs
- `SYNTAX_IMPORT_CYCLE`: Syntax/Indentation error or circular imports
- `ASSERTION_LOGIC`: Unit test failure or output mismatch
- `TIMEOUT_DEADLOCK`: Pipe buffers or lock starvation
- Injects targeted remediation capsules directly to the agent.

### 3. Dual-Stream Memory Gatekeeper (`gemma_memory_gate`)
Ensures clean separation between engineering knowledge and conversational history:
- **`stream="interaction"`**: Daily conversational chat, greetings, and memos route to the `chat` track.
- **`stream="code"`**: High-density architecture ADRs, bug post-mortems, and topologies route to `architecture` or `errors` tracks.

### 4. Codebase Audit & 800-Line Budget Gate (`gemma_post_task_audit`)
Audits workspace line budgets post-task:
- Flags any file exceeding the 800-line hard threshold (`WARN_REFACTOR`).
- Detects AST exports (classes and top-level functions).
- Generates directory tree and symbol mapping matrix ready for knowledge vault storage (`SYNC_VAULT`).

### 5. Model Tier Router (`gemma_model_route`)
Recommends optimal model tier:
- Fast routine tasks $\to$ `CLOUD_FLASH` (with `local_harness_eligible: True` for future local harness execution).
- Multi-file refactor / architecture $\to$ `CLOUD_PRO`.

---

## MCP Server Configuration

Add to your MCP configuration (e.g. `mcp_config.json`):

```json
{
  "mcpServers": {
    "gemma-prep-core": {
      "command": "python",
      "args": ["path/to/gemma-prep-core/server.py"]
    }
  }
}
```

---

## Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `GEMMA_DAEMON_URL` | Local HTTP daemon endpoint | `http://127.0.0.1:8765` |
| `GEMMA_PYTHON_EXE` | Python executable for daemon spawn | Current Python interpreter |
| `JEV_TRAIN_ROOT` | Path to Jev training/service root | `~/.gemma/jev-train` |
| `VAULT_CODE_DB` | Path to code memory SQLite database | `~/.memory-vault/vault_code.db` |
| `VAULT_INTERACT_DB` | Path to interaction SQLite database | `~/.memory-vault/vault_interaction.db` |

---

## Running Verification Tests

```bash
# Test complete modular kernel suite
python test_modular_kernel.py

# Test codebase line budget auditor
python test_post_task_audit.py
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
