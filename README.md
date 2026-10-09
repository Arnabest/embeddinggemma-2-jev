# EmbeddingGemma-2-Jev

Non-Destructive Multimodal Base with a Modular 7KB Causal Decision Adapter for Google DeepMind's `google/embeddinggemma-2`.

[English](#overview) | [中文说明](#中文概览)

---

## Overview

**EmbeddingGemma-2-Jev** upgrades Google DeepMind's `embeddinggemma-2` (744M multimodal parameter architecture) into an 8K long-context causal decision engine while preserving its native multimodal embedding capabilities.

Instead of destructive full-parameter fine-tuning, this architecture keeps 100% of the multimodal base weights frozen (`requires_grad=False`) and mounts an ultra-lightweight **7.03 KB universal decision adapter** (`adapter.pt`) over the unified 768-dimensional latent space.

### Core Primitives (Jev Framework)

- **Noul**: Binary hypothesis gating and anomaly verification (`y in {0, 1}`).
- **Choice**: Multi-option semantic and action routing with softmax confidence distributions.
- **Score**: Discrete 1-to-5 scale risk and resource status evaluation.

---

## Architectural Highlights

1. **Zero Catastrophic Forgetting**: The complete 744.4M multimodal foundation (271M text + 167M vision + 305M audio) remains untouched. Cross-modal retrieval and dense representations are 100% preserved.
2. **Modular Plug-and-Play**: The decision adapter consists of a `(3, 512)` task marker embedding and a `(768, 1)` universal scorer. The entire adapter occupies only 7.03 KB.
3. **Hardware Circuit Breaker**: Enforces native `bfloat16` precision on CUDA to eliminate Gemma-2 FP16 NaN underflow hazards. Automatically detects CUDA OutOfMemory conditions and seamlessly falls back to CPU `float32` execution without process termination.
4. **Multi-Interface Ingress**: Native support for In-Process Python SDK, REST API, stdio Model Context Protocol (MCP) server, and Antigravity / Harnessd lifecycle hooks.

---

## Project Structure

```text
embeddinggemma-2-jev/
├── README.md                      # Comprehensive documentation
├── LICENSE                        # Apache 2.0 License
├── NOTICE                         # Google DeepMind attribution
├── requirements.txt               # Pinned dependencies
├── pyproject.toml                 # Package definition
├── models/
│   ├── adapter/
│   │   ├── adapter.pt             # Pre-trained 7.03 KB decision adapter
│   │   └── adapter_meta.json      # Adapter architecture metadata
│   └── download_base_model.py     # Script to download base weights from Hugging Face
├── core/
│   ├── model.py                   # GemmaJevModel architecture
│   ├── engine.py                  # GemmaPrepEngine with Circuit Breaker
│   └── pipeline.py                # ContextAssemblyPipeline (Attention HUD)
├── serve/
│   ├── app.py                     # High-throughput REST API & Web service
│   ├── mcp_server.py              # Stdio JSON-RPC 2.0 MCP server
│   └── hook_adapter.py            # PreInvocation & PreToolUse lifecycle adapter
├── train/
│   ├── train.py                   # Modular adapter training script
│   └── distill_incremental.py     # Active learning incremental fine-tuning
├── data/
│   ├── README.md                  # Dataset schema specification
│   └── sample_dataset.jsonl       # Sanitized generic engineering samples
└── tests/
    └── test_decision_inference.py # Unit tests for inference primitives
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/Arnabest/embeddinggemma-2-jev.git
cd embeddinggemma-2-jev
pip install -r requirements.txt
```

### 2. Download Base Weights

Download the official `google/embeddinggemma-2` base weights from Hugging Face:

```bash
python models/download_base_model.py --target-dir models/base/
```

### 3. Verify Inference

Run the test suite to verify heuristic and model inferences:

```bash
python tests/test_decision_inference.py
```

---

## Usage

### In-Process Python SDK

```python
from core.engine import GemmaPrepEngine

engine = GemmaPrepEngine(
    base_model_dir="models/base",
    adapter_path="models/adapter/adapter.pt",
    device="cuda"
)

# 1. Action Routing (Choice)
decision = engine.jev_decide(
    mode="choice",
    question="Select optimal dispatch action:",
    options=[
        {"t": "Fallback to CPU float32", "d": "Prevents CUDA OOM"},
        {"t": "Force CUDA allocation", "d": "Causes system crash"}
    ],
    state="GPU VRAM occupancy reached 94% on 8GB VRAM."
)
print("Action:", decision["items"][decision["pred"]]["label"])
print("Confidence:", f"{decision['conf']:.2%}")

# 2. Safety Gating (Noul)
safety = engine.jev_decide(
    mode="noul",
    question="Detecting catastrophic data loss risk before file overwrite.",
    options=[],
    state="Target: system_config.json, Overwrite: True, Backup: False"
)
print("Is Risky:", safety["pred"] == 1)
```

### REST API Service

Launch the local HTTP service (supports long context up to 8K):

```bash
python serve/app.py --port 8765 --device cuda
```

Endpoints:
- `POST /api/predict`: Executes Jev `noul`, `choice`, or `score` primitives.
- `POST /api/intent`: Pre-instruction intent classification.
- `GET /health`: Live hardware VRAM telemetry and model status.

### Model Context Protocol (MCP) Server

To expose EmbeddingGemma-2-Jev tools to Claude, Gemini, or Antigravity agents, register the MCP server in your `mcp_config.json`:

```json
{
  "mcpServers": {
    "embeddinggemma-2-jev": {
      "command": "python",
      "args": ["serve/mcp_server.py"]
    }
  }
}
```

Exposed Tools:
- `gemma_intent_classify`: Intent classification and routing recommendations.
- `gemma_jev_decide`: Deterministic causal decision judge.
- `gemma_assemble_context`: Attention HUD context assembly.
- `gemma_hardware_status`: Live GPU memory telemetry.

---

## Training & Incremental Distillation

### Training Adapter from Scratch

To train the 7KB adapter on your custom dataset:

```bash
python train/train.py --base-model models/base --data data/sample_dataset.jsonl --epochs 3
```

### 20-Second Active Learning Update

When encountering new edge cases or error logs, update the adapter incrementally without full retraining:

```bash
python train/distill_incremental.py --base-model models/base --data data/sample_dataset.jsonl
```

---

## Data Isolation Policy

Private conversational transcripts, proprietary source code diffs, and local user trajectories are strictly isolated from this open-source repository. The `data/` directory only contains generic, sanitized engineering schemas (`sample_dataset.jsonl`).

---

## License & Attribution

- **License**: Apache License, Version 2.0 ([LICENSE](LICENSE)).
- **Base Model Attribution**: This repository builds on Google DeepMind's `google/embeddinggemma-2` architecture ([NOTICE](NOTICE)).
- **Commercial Use**: Commercial utilization, secondary modification, and closed-source application bundling are permitted under Apache 2.0 terms.

---

## 中文概览

**EmbeddingGemma-2-Jev** 是基于 Google DeepMind 开源的 `embeddinggemma-2`（744M 参数多模态模型）构建的外挂式因果决策引擎。

### 核心特性
1. **多模态底座完整保留**：744.4M 多模态权重（文本 271M + 视觉 167M + 音频 305M）完全只读冻结，杜绝灾难性遗忘。
2. **7KB 轻量决策适配器**：通过任务标记与 Universal Scorer 覆盖 768 维统一隐空间，实现 Jev 三原语（Noul 门禁二元判定、Choice 动作分流、Score 态势五级定级）。
3. **硬件容灾断路器 (Circuit Breaker)**：强制采用 `bfloat16` 消除 Gemma-2 的 FP16 数值下溢；在并发显存耗尽时自动捕获 CUDA OOM 并平滑热回退至 CPU float32。
4. **多接口生态**：提供 Python SDK、REST API、标准 stdio MCP 协议服务端与生命周期 Hook 适配器。
