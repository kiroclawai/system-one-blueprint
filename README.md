# System One Blueprint

A Jev-class "System One" model blueprint — built from open components, designed to compete head-to-head with TypeSafe's Jev on structured decision tasks.

**Status:** Blueprint / design document
**Target:** Parallel, calibrated, type-safe decision model using a 50M encoder + Needle3-inspired output heads
**License:** Apache 2.0 (matching Needle3)

---

## What this is

This is a **complete technical blueprint** for building a System One model —
a model class optimized for fast, structured, calibrated decisions rather than
free-form text generation.

The blueprint has **three implementation tiers**, from quickest to most ambitious:

| Tier | Approach | Time | What you get |
|------|----------|------|-------------|
| **Tier 1** | 50M encoder + decision heads (Option 5) | Days | Working Jev-compatible API, parallel structured output, calibrated confidence |
| **Tier 2** | Distill Tier 1 knowledge into Needle3 | Weeks | 8–29 MB model, edge-deployable, parallel decode, grammar-constrained |
| **Tier 3** | Full RLCD-style training pipeline | Months | Calibrated decision model trained end-to-end with proper scoring rules |

---

## Quick start

```bash
# Read the architecture first
cat docs/DESIGN.md

# Then training approach
cat docs/TRAINING.md

# Then what data you need
cat docs/DATASETS.md

# Then how to measure against Jev
cat docs/EVALUATION.md

# Then infrastructure
cat docs/INFRA.md
```

---

## Head-to-head competitive positioning

The goal is not to clone Jev. It's to build something that solves the same
problem (fast, structured, calibrated decisions for software automation) with
open components, at competitive speed and quality.

| Dimension | Jev (TypeSafe) | This Blueprint (Tier 1) | This Blueprint (Tier 2) |
|-----------|---------------|------------------------|------------------------|
| **Architecture** | Proprietary System One | 50M encoder + parallel heads | Needle3 SAN + decision heads |
| **Output** | Typed, parallel, calibrated | Typed, parallel, calibrated | Typed, parallel, calibrated, grammar-constrained |
| **Latency** | 70ms–500ms | ~50–200ms (encoder + heads) | ~1–10ms (quantized, edge) |
| **Cost** | $0.042/MTok input | Free (local) / API cost of encoder | Free (local, edge) |
| **Type safety** | Mathematically guaranteed | Guaranteed by head design | Guaranteed by grammar |
| **Calibration** | RLCD-trained, calibrated | Post-hoc + loss-based calibration | Distillation + fine-tuning calibration |
| **Footprint** | Cloud API | ~200MB (encoder + runtime) | 8–29 MB (quantized .cact) |
| **Deployment** | API only | API server + local | Edge, mobile, air-gapped |
| **Training data** | Proprietary 360B+ tokens | Open datasets + synthetic | Distilled from Tier 1 + open data |
| **Open weights** | No | Encoder: yes (open) | Full model: yes (Apache 2.0) |

---

## Directory structure

```
system-one-blueprint/
├── README.md              # This file
├── docs/
│   ├── DESIGN.md          # Architecture blueprint (Tier 1/2/3)
│   ├── TRAINING.md        # Training pipeline (distillation, RLCD-style, calibration)
│   ├── DATASETS.md        # Open datasets + synthetic data construction
│   ├── EVALUATION.md      # Head-to-head eval methodology vs Jev
│   └── INFRA.md           # Infrastructure, deployment, costs
├── src/
│   ├── encoder_heads/     # Tier 1 reference implementation (encoder + decision heads)
│   ├── grammar_decode/    # Grammar-constrained output decoding
│   └── calibration/       # Calibration layers and metrics
├── evals/
│   ├── workflows/         # Workflow evaluation harness (TypeSafe-style)
│   └── benchmarks/        # Standard benchmarks for comparison
├── data/
│   └── construction/      # Scripts for building training data from open sources
└── infra/
    └── deployment/        # API server, edge deployment, quantization
```

---

## Documents

| Document | Description |
|----------|-------------|
| [`docs/DESIGN.md`](docs/DESIGN.md) | Full architecture: encoder choice, head design, parallel output, grammar constraints, calibration |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Three-tier training pipeline: supervised heads → distillation → RLCD-style calibration |
| [`docs/DATASETS.md`](docs/DATASETS.md) | Open dataset inventory + construction recipes for (state, questions, decisions) triples |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | How to measure head-to-head against Jev: workflow evals, calibration metrics, speed, cost |
| [`docs/INFRA.md`](docs/INFRA.md) | Infrastructure requirements per tier, deployment options, cost model |

---

## Why this approach

The key insight from both Jev and Needle3:

> **If you give up free-form text generation, you gain speed, type safety, and
> calibration that autoregressive models cannot match.**

Jev does this with a proprietary architecture and RLCD training.
Needle3 does this with a Laddered Simple Attention Network for tool-calling.

This blueprint does it with:
1. A **50M open encoder** (Battle-tested, widely available — MiniLM, BERT-tiny, T5-small)
2. **Parallel decision heads** (one per question type: Yes/No, Choice, Score)
3. **Grammar-constrained output** (output space defined by schema, no hallucination possible)
4. **Calibration-aware training** (Brier score / proper scoring rules in the loss)
5. **Needle3 distillation** (Tier 2: collapse into 8–29 MB quantized model)

The result is a model that:
- Takes unstructured text + typed questions → returns typed decisions with calibrated confidence
- Runs in parallel (all questions answered in one forward pass)
- Cannot produce type errors (output space is schema-constrained)
- Is small enough to run on edge devices (Tier 2)
- Uses entirely open components

---

## Contributing

This is a living blueprint. The design documents describe the target architecture;
the reference implementation in `src/` provides working code for Tier 1.

See individual documents for detailed design decisions and trade-offs.
