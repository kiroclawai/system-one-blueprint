# Infrastructure — System One Model

## Overview

Infrastructure requirements for each tier, deployment options, and cost models.

---

## Tier 1: Encoder + Heads

### Development environment

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | None (CPU training works for small models) | 1× RTX 3090 / 4090 (24GB) or A100 (40/80GB) |
| **CPU RAM** | 16 GB | 32 GB |
| **Disk** | 10 GB free | 50 GB free |
| **Python** | 3.9+ | 3.10+ |
| **Key libraries** | PyTorch, Transformers, NumPy, Scikit-learn | Same + Weights & Biases (experiment tracking) |

**Training on CPU:** Possible for Tier 1 with a frozen encoder. Training 10K examples with MiniLM (22M) + heads takes hours on a modern CPU. Not recommended for iteration speed, but viable for final training runs.

**Training on GPU:** Recommended. A single consumer GPU (RTX 3090/4090) can train Tier 1 in minutes to hours depending on data volume.

### Inference environment

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Modern x86_64 (Ryzen 5 / Core i5 or better) | Modern x86_64 (Ryzen 7 / Core i7) |
| **GPU** | None required | Optional: any GPU for faster inference |
| **RAM** | 4 GB | 8 GB |
| **Network** | Any | Low-latency connection for API serving |

**Inference on CPU:** The encoder (22M params) + heads is small enough to run on CPU at acceptable speed. MiniLM inference on a modern CPU processes a 500-token input in ~50–100ms. Total decision latency: 50–200ms.

**Inference on GPU:** 10–50ms total latency. Only worth it if you need high throughput or sub-50ms latency.

### API server

```python
# Minimal FastAPI server for Tier 1
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()
model = load_system_one_model("best_model.pt")
model.eval()

class DecisionRequest(BaseModel):
    state: str
    questions: list[dict]

class DecisionResponse(BaseModel):
    decisions: dict
    latency_ms: float

@app.post("/v1/decide", response_model=DecisionResponse)
async def decide(request: DecisionRequest):
    import time
    start = time.time()
    
    with torch.no_grad():
        outputs = model([request.state], request.questions)
    
    decisions = construct_output(request.questions, outputs)
    latency_ms = (time.time() - start) * 1000
    
    return DecisionResponse(decisions=decisions, latency_ms=latency_ms)
```

**Serving options:**
- **FastAPI** (above) — simple, Python-native, good for development and low-traffic production
- **vLLM** — if you need high throughput (not really needed for decision models, but available)
- **Docker container** — package the model + server for deployment

### Scaling

Tier 1 inference is lightweight. A single CPU server can handle:
- **Low traffic:** 100–1000 queries/day on a $5/month VPS
- **Medium traffic:** 10K–100K queries/day on a $20–50/month server (with batching)
- **High traffic:** 1M+ queries/day requires multiple instances + load balancing

For high-traffic deployments, consider:
- **Batching:** Process multiple queries in a single forward pass
- **Horizontal scaling:** Run multiple instances behind a load balancer
- **GPU acceleration:** If CPU becomes the bottleneck

---

## Tier 2: Needle3 Distillation

### Training environment

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 1× RTX 3090 (24GB) | 1× A100 (40GB) or better |
| **CPU RAM** | 32 GB | 64 GB |
| **Disk** | 50 GB free | 200 GB free |
| **Needle3 SDK** | cactus-needle Python package | Latest from Cactus Compute |

**Needle3 fine-tuning:** The Cactus Platform handles the 2-bit post-training quantization. For local fine-tuning, use the `cactus-needle` Python package with LoRA.

**Disk space for distillation data:** 100K examples at ~1KB each = ~100MB. 1M examples = ~1GB. Not a concern.

### Inference environment (Needle3 engine)

| Platform | Engine size | Model size (CQ2) | Speed |
|----------|------------|------------------|-------|
| **Linux x86_64** | <1 MB | 8–29 MB | 1–10K tokens/s prefill |
| **macOS (Apple Silicon)** | <1 MB | 8–29 MB | Similar to Linux |
| **Windows** | <1 MB | 8–29 MB | Similar to Linux |
| **Raspberry Pi 5** | <1 MB | 8–29 MB | 400–4K tokens/s decode |
| **WebAssembly (browser)** | WASI component | 8–29 MB | Varies |
| **Mobile (ARM)** | <1 MB | 8–29 MB | Varies |

**Key advantage:** The Needle3 engine is <1 MB and runs anywhere. The model is 8–29 MB (CQ2 quantized). Combined footprint: ~10–30 MB. This runs on a Raspberry Pi, an Android phone, an air-gapped industrial controller, or in a browser WebAssembly module.

**No GPU required.** Needle3 is designed for tiny devices. The quantized model runs on CPU-only hardware.

### Deployment

**Local/binaries:**
```bash
# Download engine + model
./needle --model system-one-8l.cact --tools questions.json --prompt "state text here"

# Start API server
./needle --model system-one-8l.cact --tools questions.json --serve
```

**Embedded:**
```c
// C API (air-gapped, embedded systems)
#include "needle3.h"

needle_model_t* model = needle_load("system-one-8l.cact");
needle_init(model, tools_json, tools_json_len);
needle_response_t* response = needle_run(model, state_text, state_text_len);
// Process response...
needle_free(response);
needle_unload(model);
```

**Docker (if needed):**
```dockerfile
FROM debian:bookworm-slim
COPY needle3-engine /usr/local/bin/
COPY system-one-8l.cact /app/
COPY questions.json /app/
EXPOSE 8080
CMD ["needle", "--model", "/app/system-one-8l.cact", "--tools", "/app/questions.json", "--serve"]
```

---

## Tier 3: RLCD Training

### Training environment

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU cluster** | 1× A100 (40GB) | 4× A100/H100 |
| **CPU RAM** | 64 GB | 256 GB |
| **Disk (model checkpoints)** | 100 GB | 1 TB |
| **Disk (training data)** | 100 GB | 1 TB |
| **Network** | 1 Gbps | 10 Gbps (for multi-node) |

**RL training is significantly more demanding** than supervised training. Expect to need:
- More GPUs (the RL loop is less efficient than standard backprop)
- More data (RL needs exploration, which means more samples)
- More time (RLCD-style training may take days to weeks)

**If you don't have GPU clusters:** Tier 3 is not feasible without cloud GPU access. Consider:
- **Cloud training:** Run on AWS (p4d instances), Lambda Labs, RunPod, or similar
- **Skip Tier 3:** Focus on Tier 1 + Tier 2, which are much more practical
- **Collaborate:** Partner with someone who has GPU resources

### Estimated training costs (cloud)

| Tier | GPU | Hours | Instance cost/hr | Total |
|------|-----|-------|-----------------|-------|
| **Tier 1** | RTX 3090 equivalent | 2–10 hrs | $0.50–2.00 | $1–20 |
| **Tier 2** | A100 40GB | 10–50 hrs | $1.00–3.00 | $10–150 |
| **Tier 3** | 4× A100 | 100–500 hrs | $4.00–12.00 | $400–6000 |

These are rough estimates. Actual costs depend on data volume, model size, training efficiency, and cloud provider pricing.

---

## Deployment comparison

| Deployment aspect | Tier 1 (encoder + heads) | Tier 2 (Needle3) | Tier 3 (custom) |
|------------------|--------------------------|------------------|-----------------|
| **Runtime** | PyTorch / HF / ONNX | Needle3 engine (<1 MB) | Custom or PyTorch |
| **Model size** | ~200 MB (encoder + heads + tokenizer) | 8–29 MB (CQ2 quantized) | Varies |
| **GPU required** | No (CPU works) | No (CPU works) | No (but training needs GPU) |
| **Edge deployable** | Yes (but heavier) | Yes (designed for it) | Depends on architecture |
| **Mobile deployable** | Possible (ONNX/TFLite) | Yes (ARM binary) | Harder |
| **Browser deployable** | Via ONNX/WebGPU | Via WASM | Via ONNX/WebGPU |
| **Air-gapped** | Yes | Yes | Yes |
| **API server** | FastAPI / vLLM | Needle3 --serve | Custom |
| **Batching** | Possible | Engine-dependent | Custom |
| **Cold start** | Seconds (model load) | Milliseconds (memory-map .cact) | Varies |

---

## Cost model

### Self-hosted vs. Jev

**Jev pricing (from their site):**
- $0.042 per million input tokens
- Output tokens: free
- 5 free runs for new accounts

**Example cost comparison for 1 million queries:**

Assumptions: 500 input tokens per query, 4 questions per query.

| | Jev | Tier 1 (self-hosted) | Tier 2 (self-hosted) |
|---|-----|---------------------|---------------------|
| **Compute cost** | $0.042 × 1M × 500 / 1M = **$21** | **$0** (already own hardware) | **$0** |
| **Hardware cost** | $0 (API) | $5–50/month (VPS) | $5–50/month (VPS) |
| **Development cost** | $0 (use API) | $1–20 (training) | $10–150 (training) |
| **Total first month** | $21 | $6–70 | $15–200 |
| **Total subsequent months** | $21/month | $5–50/month | $5–50/month |

**At scale (100M queries/month):**
| | Jev | Tier 1/2 self-hosted |
|---|-----|---------------------|
| **Monthly cost** | $0.042 × 100M × 500 / 1M = **$2,100** | **$50–200** (larger server) |
| **Annual saving** | — | **$22,600–24,480/year** |

The cost advantage of self-hosting grows with volume. At low volume, Jev's pricing is competitive (5 free runs, cheap per-token pricing). At high volume, self-hosting wins significantly.

### When to use Jev vs. self-hosted

**Use Jev when:**
- You're prototyping and don't want to build infrastructure
- You need the best possible calibration (RLCD-trained) and can't replicate it
- Your volume is low (free tier + cheap pricing is hard to beat)
- You don't have GPU resources for training
- You want to evaluate Jev against your own needs before committing

**Self-host Tier 1 when:**
- You have a clear use case and domain
- You want full control over the model and data
- You want to avoid API dependencies and vendor lock-in
- Your volume is moderate (self-hosting cost < Jev cost)
- You want to customize the question types and decision logic

**Self-host Tier 2 when:**
- You need edge deployment (no network, air-gapped, low-latency)
- You need the smallest possible footprint (8–29 MB)
- You want to run on devices (mobile, Raspberry Pi, embedded)
- Your volume is high enough to justify training cost
- You want the Needle3 engine benefits (speed, efficiency, grammar constraints)

**Self-host Tier 3 when:**
- You have GPU resources and ML engineering capacity
- You want to push the state of the art in calibrated decision models
- You need custom architecture for specific requirements
- The investment is justified by the application's value

---

## Security and privacy

### Data handling

**Tier 1/2 self-hosted:**
- All data stays on your infrastructure
- No data sent to external APIs (except during training if you use cloud GPUs)
- Full control over data retention, logging, compliance

**Jev API:**
- Data sent to TypeSafe's API (hosted on West Coast)
- Check TypeSafe's privacy policy for data retention and usage
- May not be suitable for sensitive/confidential data

### Air-gapped deployment

Tier 2 (Needle3) is designed for air-gapped environments:
- Single .cact file + engine binary
- No network access needed
- Runs entirely locally

This is critical for:
- Industrial control systems
- Medical devices
- Defense/government applications
- Financial trading systems
- Any environment where network access is restricted or undesirable

---

## Observability

### Logging

Log every decision with:
- Input state (or hash of state for privacy)
- Questions asked
- Decisions made (answers + probabilities + confidence)
- Latency
- Model version

```python
import json
import time
from datetime import datetime

def log_decision(request, response, model_version):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "state_hash": hash(request.state),  # or store full state if appropriate
        "questions": request.questions,
        "decisions": response.decisions,
        "latency_ms": response.latency_ms,
    }
    # Write to structured log (JSON Lines format)
    with open("decisions.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

### Monitoring

Track:
- **Decision distribution:** Are certain options being chosen much more than expected?
- **Confidence distribution:** Are confidences systematically high or low?
- **Latency:** P50, P90, P99 latency over time
- **Error rate:** Type errors (should be zero), timeout errors, etc.
- **Calibration drift:** ECE over time — is calibration degrading?

### Feedback loop

Collect outcomes (whether decisions were correct) to improve the model over time:

```python
def collect_outcome(decision_id, actual_outcome):
    """
    Store the actual outcome for a decision.
    Use this to:
    1. Measure real-world accuracy
    2. Retrain/calibrate the model
    3. Identify failure patterns
    """
    store_outcome(decision_id, actual_outcome)
```

This is essential for Tier 3 (RLCD training needs outcome data) and valuable for Tier 1/2 calibration improvement.

---

## Summary

| Need | Recommended tier | Infrastructure |
|------|-----------------|----------------|
| **Prototype / evaluate** | Tier 1 (or even just GPT-4o with structured output) | Laptop / free tier cloud |
| **Production API** | Tier 1 | Single server (CPU or GPU) |
| **Edge / mobile / air-gapped** | Tier 2 | Needle3 engine + .cact model |
| **Maximum performance** | Tier 3 | GPU cluster for training, then deploy like Tier 1/2 |
| **Low volume, minimal effort** | Jev API | No infrastructure needed |

---

## Quick reference: what you need to start

**To build Tier 1 (fastest path to a working System One model):**

1. A machine with Python 3.10+, 16GB RAM, 10GB disk (no GPU required)
2. `pip install torch transformers scikit-learn fastapi uvicorn`
3. Download `sentence-transformers/all-MiniLM-L6-v2` (auto-downloaded on first use)
4. Construct 10K–100K training examples (see DATASETS.md)
5. Train heads (hours on CPU, minutes on GPU)
6. Deploy with FastAPI

**Total time:** 1–3 days for a working prototype.
**Total cost:** $0–20 (if using cloud GPU for training).

**To build Tier 2 (edge-deployable):**

1. Everything for Tier 1, plus:
2. Cactus Compute account / Needle3 SDK access
3. GPU for distillation training (or cloud GPU)
4. Export to .cact format
5. Deploy with Needle3 engine

**Total time:** 1–3 weeks.
**Total cost:** $10–200 (cloud GPU for distillation).

**To build Tier 3 (RLCD from scratch):**

1. GPU cluster (1–4 A100/H100)
2. Significant ML engineering capacity
3. Large training data (100K–10M+ examples)
4. Custom RL training infrastructure
5. Months of development and experimentation

**Total time:** Months.
**Total cost:** $500–10,000+ (cloud GPU training).
