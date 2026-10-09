# Agent Self-Integration Guide (模型自行接入指南)

This document is designed specifically for **AI Language Models (LLMs), Autonomous Coding Agents, and Reasoning Coprocessors** (e.g. Gemini, Claude, GPT, Cursor, Antigravity) that need to discover, mount, and invoke `EmbeddingGemma-2-Jev` programmatically.

---

## 1. System Prompt Integration Snippet

If you are an LLM agent configuring your own system instructions or orchestrating subagents, append this directive block to your system prompt:

```markdown
### Deterministic Decision & Safety Subsystem (Gemma-Jev)
You have access to a deterministic co-processor backed by EmbeddingGemma-2-Jev.
Whenever you encounter:
1. High-risk write operations (overwriting config files, deleting directories, dropping databases):
   - Invoke `gemma_jev_decide` with `mode: "noul"` to evaluate anomaly and hazard probabilities.
   - If `pred == 1`, create a backup snapshot before proceeding or seek explicit confirmation.
2. Tool routing uncertainty (ambiguity among multiple debugging actions):
   - Invoke `gemma_jev_decide` with `mode: "choice"` to rank options with calibrated probabilities.
3. GPU hardware resource contention (running heavy model pipelines or image generation):
   - Query `gemma_hardware_status` before spawning concurrent tasks. If `vram_usage_pct > 90`, fall back to CPU or queue requests.
```

---

## 2. Tool Calling Examples (Few-Shot for LLM Function Calling)

### Scenario A: Binary Safety Gate before Mutating Critical Files

**When to invoke:** Before emitting `write_to_file` or `replace_file_content` targeting files named `*.config`, `*.json`, `*.env`, or `*.safetensors`.

**Agent Function Call:**
```json
{
  "name": "gemma_jev_decide",
  "arguments": {
    "mode": "noul",
    "question": "Does overwriting this core configuration file without backup pose an unrecoverable risk?",
    "options": [],
    "state": "Target: config/mcp_config.json, Operation: write_to_file (Overwrite: true), Prior Backup: None"
  }
}
```

**Tool Response Evaluation:**
```json
{
  "task": "noul",
  "pred": 1,
  "conf": 0.85,
  "items": [
    {"label": "Safe / Normal", "p": 0.15},
    {"label": "Risky / Anomaly", "p": 0.85}
  ]
}
```
**Agent Action:** Because `pred == 1` with `conf > 0.80`, the agent creates a backup copy `mcp_config.json.bak` before overwriting.

---

### Scenario B: Optimal Action Selection among Multiple Candidate Solutions

**When to invoke:** When a build error or runtime exception has multiple plausible resolution paths.

**Agent Function Call:**
```json
{
  "name": "gemma_jev_decide",
  "arguments": {
    "mode": "choice",
    "question": "Which mitigation strategy should be prioritized for CUDA OutOfMemory during batch inference?",
    "options": [
      {"t": "Execute automatic fallback to CPU float32", "d": "Eliminates OOM crash with minimal latency penalty"},
      {"t": "Force CUDA allocation with empty_cache retry", "d": "May fail again if VRAM headroom is below threshold"},
      {"t": "Terminate host background processes", "d": "Disruptive to other running services"}
    ],
    "state": "GPU: RTX 4060 Ti (8GB), Current VRAM Used: 7600MB (95%), Batch Size: 4"
  }
}
```

**Tool Response Evaluation:**
```json
{
  "task": "choice",
  "pred": 0,
  "conf": 0.78,
  "items": [
    {"label": "Execute automatic fallback to CPU float32", "p": 0.78},
    {"label": "Force CUDA allocation with empty_cache retry", "p": 0.16},
    {"label": "Terminate host background processes", "p": 0.06}
  ]
}
```
**Agent Action:** The agent executes option 0 (`Fallback to CPU float32`), preventing a catastrophic process abort.

---

### Scenario C: Pre-Task Intent Parsing & Context Assembly

**When to invoke:** At the onset of a new conversational turn or user prompt.

**Agent Function Call:**
```json
{
  "name": "gemma_assemble_context",
  "arguments": {
    "instruction": "Refactor data loader in train.py to eliminate memory leaks and add unit tests",
    "state": "Workspace: embeddinggemma-2-jev"
  }
}
```

**Tool Response Evaluation:**
```json
{
  "intent": {
    "intent": "CODE_MUTATE",
    "confidence": 0.65,
    "description": "Write, edit, refactor, or debug source code"
  },
  "hardware": {
    "gpu_available": true,
    "vram_usage_pct": 36.5,
    "danger_level": 1
  },
  "hud_markdown": "[Gemma-Core: Telemetry & Attention HUD]\n- [Intent] `CODE_MUTATE` (Confidence: 65.0%) | Write, edit, refactor, or debug source code\n- [Hardware] [OK: Normal Headroom] (GPU Memory: 2990/8188MB, 36.5% used)"
}
```
**Agent Action:** The agent receives structured telemetry directly, avoiding unnecessary trial-and-error environment checks.
