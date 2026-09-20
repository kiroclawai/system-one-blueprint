# Architecture Blueprint — System One Model

## Overview

A System One model takes **unstructured text state** + **typed decision questions** and returns **typed probabilistic decisions in parallel**, with calibrated confidence on every output.

This document describes the architecture across three tiers:

| Tier | Description | Key components |
|------|-------------|----------------|
| **Tier 1** | 50M open encoder + parallel decision heads | Pretrained encoder (MiniLM/BERT-tiny/T5-small) + per-question-type MLP heads + calibration layer |
| **Tier 2** | Distill into Needle3 architecture | Needle3 SAN backbone + decision heads + grammar-constrained decoding + quantization |
| **Tier 3** | RLCD-style end-to-end training | Custom architecture trained from scratch with calibration-aware RL |

---

## Tier 1: Encoder + Decision Heads (Reference Implementation)

### Design principle

**Keep the text understanding from a proven open encoder. Replace the generative decoder with parallel structured output heads.**

An autoregressive LLM spends most of its compute generating tokens one at a time. For structured decisions, none of that is needed — you want a single forward pass that produces a decision vector for each question.

### Architecture diagram

```
Input: state_text (string/JSON) + question_specs (list of typed questions)
│
├── State encoding Path
│   │
│   └── [Pretrained Encoder] ← 50M open model (frozen or lightly tuned)
│           │
│           └── hidden_state: [batch, seq_len, hidden_dim]
│
├── Question encoding Path
│   │
│   └── [Question Spec Encoder] ← embeds each question's type + options + scale
│           │
│           └── question_embeddings: [num_questions, q_embed_dim]
│
├── Cross-attention / pooling
│   │
│   └── [State-Question Interaction] ← how does the state relate to each question?
│           │
│           └── pooled representations: [num_questions, hidden_dim]
│
└── Output Heads (one per question, all run in parallel)
    │
    ├── Head_1 (Yes/No) ──→ P(yes), P(no) + confidence
    ├── Head_2 (Choice[4]) ──→ P(opt_1), P(opt_2), P(opt_3), P(opt_4) + confidence
    ├── Head_3 (Score[likert]) ──→ value + confidence
    ├── Head_4 (Yes/No) ──→ P(yes), P(no) + confidence
    └── ...
```

### Component details

#### 1. Pretrained Encoder

**Selection criteria:**
- ~50M parameters (or smaller)
- Strong text understanding / sentence embedding quality
- Open weights (Apache 2.0, MIT, or similar permissive license)
- Single forward pass produces good representations
- Available on HuggingFace with safetensors

**Candidates (ranked by practicality):**

| Model | Params | Dim | License | Downloads | Recommendation |
|-------|--------|-----|---------|-----------|----------------|
| `sentence-transformers/all-MiniLM-L6-v2` | ~22M | 384 | MIT | 252M | **Primary choice** — best quality/params, excellent embeddings, fast |
| `prajjwal1/bert-tiny` | ~4.4M | 128 | Apache 2.0 | 950K | Ultra-small fallback — if 22M is too much |
| `google-t5/t5-small` | ~60M | 512 | Apache 2.0 | 24.8M | Encoder-decoder — more flexible but heavier |
| `axolotl-ai-co/tiny-llama-50m` | ~50M | 512 | Unknown | 10K | Causal LM — less ideal for encoding, more for generation |
| `distilbert/distilbert-base-uncased` | ~66M | 768 | Apache 2.0 | 7.6M | Higher quality but over 50M target |

**Decision:** Start with `all-MiniLM-L6-v2` (22M, 384 dim). If capacity is insufficient, move to `t5-small` (60M) or `distilbert-base` (66M). The 50M target is a guideline — the right encoder is the one that gives good decision accuracy at acceptable speed.

**Training mode for encoder:**
- **Frozen (recommended start):** Encoder weights frozen, only heads trained. Fast training, less data needed, less risk of catastrophic forgetting.
- **Lightly tuned:** Encoder with low learning rate + LoRA. Better adaptation to decision domain, more training time.
- **Full fine-tune (Tier 3):** Encoder fully trained with calibration-aware RL. Maximum performance, maximum complexity.

#### 2. Question Spec Encoder

Each question has a type and parameters:

```python
QuestionSpec = {
    "id": "team",                          # unique question ID
    "type": "choice",                      # "yes_no" | "choice" | "score"
    "options": ["billing", "technical", "sales", "account"],  # for choice
    "scale": ["calm", "mildly_annoyed", "frustrated", "angry_churn"],  # for score
    "description": "Which team should handle this ticket?",  # natural language
}
```

The Question Spec Encoder converts each spec into a fixed-dimensional embedding:

```python
def encode_question_spec(spec: QuestionSpec, text_encoder) -> Tensor:
    """
    Encode a question specification into a vector.
    
    Approach: use the text encoder to encode the natural language description,
    then augment with type/options embeddings.
    """
    # 1. Encode the description text
    desc_emb = text_encoder.encode(spec["description"])  # [dim]
    
    # 2. Type embedding (learned lookup)
    type_emb = type_embedding_table[spec["type"]]  # [dim]
    
    # 3. Options/scale embedding (pooled from encoded option texts)
    if spec["type"] == "choice":
        option_embs = [text_encoder.encode(opt) for opt in spec["options"]]
        option_emb = mean_pool(option_embs)  # [dim]
    elif spec["type"] == "score":
        option_emb = mean_pool([text_encoder.encode(s) for s in spec["scale"]])
    else:  # yes_no
        option_emb = text_encoder.encode("yes or no")  # [dim]
    
    # 4. Combine
    combined = linear(desc_emb + type_emb + option_emb)  # [dim]
    return combined
```

**Simpler alternative:** Just encode the full prompt "Question: {description} Options: {options}" with the text encoder and use that as the question embedding. This avoids the custom spec encoder but loses structure.

**Recommendation:** Start with the simpler approach (encode the full prompt). Move to the structured spec encoder if you need better generalization to unseen question types.

#### 3. State-Question Interaction

The encoder produces a sequence of hidden states for the input text. For each question, we need a single pooled representation that captures how the state relates to that question.

**Pooling strategies (in order of complexity):**

**a. CLS pooling (simplest):**
```python
# Use the [CLS] token (or mean pool) as the state representation
state_repr = encoder_output.last_hidden_state[:, 0, :]  # [batch, dim]
# For each question, concatenate with question embedding and pass through head
per_question_repr = torch.cat([state_repr, question_emb], dim=-1)
```

**b. Attention pooling (better):**
```python
# Use the question embedding as a query to attend over state tokens
attn_scores = torch.bmm(question_emb, state_tokens.transpose(1, 2))  # [batch, 1, seq]
attn_weights = softmax(attn_scores, dim=-1)
pooled = torch.bmm(attn_weights, state_tokens)  # [batch, 1, dim]
```

**c. Cross-attention layer (best, most expensive):**
Add a lightweight cross-attention layer between encoder output and question embeddings. This lets each question "look at" the parts of the state most relevant to it.

**Recommendation:** Start with CLS pooling + concatenation (a). If accuracy is insufficient, move to attention pooling (b). Cross-attention (c) is only needed for complex multi-hop reasoning on long state texts.

#### 4. Output Heads

Each question type gets its own head. All heads are simple MLPs (1–2 layers) that take the pooled state-question representation and produce a structured output.

**Yes/No head:**
```python
class YesNoHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 2),  # yes, no logits
        )
    
    def forward(self, x):
        logits = self.net(x)  # [batch, 2]
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values  # confidence = max probability
        return {"probabilities": probs, "confidence": confidence}
```

**Choice head (K options):**
```python
class ChoiceHead(nn.Module):
    def __init__(self, input_dim, num_options):
        super().__init__()
        self.num_options = num_options
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_options),
        )
    
    def forward(self, x):
        logits = self.net(x)  # [batch, K]
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        return {"probabilities": probs, "confidence": confidence}
```

**Score head (ordinal/Likert):**
```python
class ScoreHead(nn.Module):
    def __init__(self, input_dim, num_levels):
        super().__init__()
        self.num_levels = num_levels
        # Treat as classification over ordered levels
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_levels),
        )
    
    def forward(self, x):
        logits = self.net(x)  # [batch, num_levels]
        probs = torch.softmax(logits, dim=-1)
        # Expected value as the score
        level_indices = torch.arange(self.num_levels, device=x.device).float()
        expected_value = (probs * level_indices).sum(dim=-1)
        confidence = probs.max(dim=-1).values
        return {"expected_value": expected_value, "probabilities": probs, "confidence": confidence}
```

**Head design notes:**

- All heads output **probabilities over the valid output space** — this is what makes the output type-safe. The model literally cannot output something outside the defined options.
- Confidence = max probability. This is a simple default; better calibration (Tier 3) replaces this with a learned calibration layer.
- Heads are independent — they all run in parallel from the same pooled representation. No sequential dependency.
- Total head parameters: negligible compared to encoder (~0.1–1M total for 10–20 questions).

#### 5. Grammar-Constrained Output Decoding

Even though the heads output constrained probabilities, the **final output must be valid JSON matching the question schema**. This is the "grammar" layer.

**Why it matters:** The model's raw output could be post-processed incorrectly. The grammar layer guarantees the final output is valid.

**Implementation options:**

**a. Direct construction (simplest, recommended for Tier 1):**
Since the heads already produce valid probability distributions over valid options, the output is constructed directly:
```python
def construct_output(questions, head_outputs):
    result = {}
    for q, out in zip(questions, head_outputs):
        if q["type"] == "yes_no":
            yes_prob = out["probabilities"][1].item()  # index 1 = "yes"
            result[q["id"]] = {
                "type": "yes_no",
                "answer": "yes" if yes_prob > 0.5 else "no",
                "probability": yes_prob,
                "confidence": out["confidence"].item(),
            }
        elif q["type"] == "choice":
            probs = out["probabilities"]
            best_idx = probs.argmax().item()
            result[q["id"]] = {
                "type": "choice",
                "answer": q["options"][best_idx],
                "probabilities": {opt: probs[i].item() for i, opt in enumerate(q["options"])},
                "confidence": out["confidence"].item(),
            }
        elif q["type"] == "score":
            result[q["id"]] = {
                "type": "score",
                "answer": q["scale"][out["expected_value"].int().item()],
                "value": out["expected_value"].item(),
                "confidence": out["confidence"].item(),
            }
    return result
```

No parsing needed — the output is constructed to be valid by design.

**b. Grammar-constrained decoding library (for Tier 2/3):**
Use `outlines` or `guidance` to constrain generation. This is relevant when the model generates text (e.g., the reasoning trace or a natural language explanation alongside the decision).

**c. Byte-level grammar (Needle3-style, for Tier 2):**
Needle3 uses a byte-level grammar compiled from the schema. Every token the model generates is guaranteed to be valid. This is the strongest form of type safety.

**Recommendation:** Tier 1 uses direct construction (a) — it's simple, correct, and fast. Tier 2 adopts Needle3's byte-level grammar approach.

#### 6. Calibration Layer

**The calibration problem:** Neural network probabilities are typically overconfident. A model that says "95% confident" might only be right 70% of the time. For automation, this is unacceptable — you need to know when the model is uncertain.

**Calibration approaches (increasing complexity):**

**a. Intrinsic confidence (Tier 1 default):**
Use `confidence = max(probabilities)` directly. Simple but typically overconfident. Works OK for clear-cut decisions, poor for ambiguous ones.

**b. Temperature scaling (Tier 1 calibration):**
Learn a single temperature parameter T that divides the logits before softmax:
```python
def calibrated_softmax(logits, temperature):
    return torch.softmax(logits / temperature, dim=-1)
```
Learned on a validation set by minimizing NLL or ECE. Simple, effective, one parameter per head.

**c. Isotonic regression (Tier 1+ calibration):**
Fit a monotonic function mapping raw confidence → actual accuracy on a held-out set. More flexible than temperature scaling, requires more data.

**d. Learned calibration head (Tier 2+):**
Add a small head that takes the raw logits and produces calibrated probabilities. Trained with a proper scoring rule (Brier score, log loss) as part of the loss function.

**e. RLCD-style calibration (Tier 3):**
The full TypeSafe approach: train with a reward that includes both correctness and calibration. The model learns to be confidently right and appropriately uncertain. This is the gold standard but requires RL infrastructure.

**Recommendation:** Start with temperature scaling (b) on top of intrinsic confidence (a). This gives you decent calibration with one learned parameter per head. If you need better, move to learned calibration head (d).

---

## Tier 2: Needle3 Distillation

### Design principle

**Take the knowledge from Tier 1's encoder+heads system and compress it into Needle3's efficient architecture.**

Needle3's architecture (Laddered Simple Attention Network) provides:
- **8–29 MB** quantized models (CQ2-bit)
- **Parallel decode** with grammar-constrained output
- **Calibrated confidence** from a learned head
- **Ladder structure** — any depth 2–20 is deployable
- **Edge deployment** — runs on Raspberry Pi 5, mobile, wearables

### Distillation pipeline

```
Tier 1 system (encoder + heads)
    │
    ├── Label large dataset ──→ (state, questions, decisions, confidence) triples
    │
    └── Tier 2: Train Needle3 on labeled data
            │
            ├── Supervised distillation: match Tier 1 outputs
            │   └── Loss = KL(needle3_probs || tier1_probs) + MSE(needle3_conf || tier1_conf)
            │
            ├── Calibration fine-tuning: improve confidence quality
            │   └── Loss = Brier_score(needle3_probs, actual_outcomes)
            │
            └── Grammar-constrained decode: compile schema → byte-level grammar
                └── Needle3's existing grammar system handles this
```

### Needle3 adaptation for decisions

Needle3's existing heads are designed for tool-calling and extraction. For general decisions, you need:

1. **New output heads** — Replace or supplement tool-calling heads with Yes/No, Choice, and Score heads matching Tier 1's design.

2. **Question conditioning** — Needle3 conditions on tool schemas. You need equivalent conditioning on decision question specs. The mechanism is similar: the schema becomes part of the input context, and the model's constrained decode ensures valid output.

3. **Confidence head** — Needle3 already has a calibrated confidence head. Verify it transfers to decision tasks or retrain it.

4. **Ladder selection** — For decision tasks, you may not need the full 20-layer model. Start with 4–8 layers (29–66 MB before quantization, 8–15 MB after CQ2). Fine-tune at full depth, then slice to the desired depth.

### Quantization

Needle3 uses **CQ2 (Cactus Quants)** at 2.125 bits per weight. The shipped model is 8–29 MB for 2–20 layers.

For your distilled model:
1. Train at full precision (float32 or bfloat16)
2. Apply LoRA fine-tuning if needed
3. Merge adapter into base
4. Export to .cact format at desired depth (2–20 layers)
5. CQ2 quantize

The quantized model runs on Needle3's engine (<1 MB runtime) on any supported platform: Linux, macOS, Windows, mobile, WebAssembly, air-gapped.

---

## Tier 3: RLCD-Style End-to-End Training

### Design principle

**Train a custom architecture end-to-end with a calibration-aware reinforcement learning objective.**

This is the most ambitious tier and the closest to what TypeSafe does with RLCD. It's also the least defined — RLCD is proprietary and the exact algorithm is not public.

### What we know about RLCD (from TypeSafe's blog)

1. **Verifiable rewards** — outputs that can be programmatically checked, not human raters
2. **Calibrated decisions** — the model is rewarded for being appropriately confident, not just correct
3. **Parallel output** — all decisions in one forward pass
4. **Different from RLHF** — RLHF optimizes for human preference on chat; RLCD optimizes for calibrated decision quality
5. **Different from RLVR** — RLVR (DeepSeek-R1 style) uses verifiable rewards but still generates text autoregressively; RLCD generates structured decisions in parallel

### Inferring the RLCD algorithm (speculative)

Based on the research literature and TypeSafe's description, RLCD likely involves:

**Reward function:**
```
R(decision, truth) = accuracy_bonus(decision, truth) + calibration_bonus(decision, truth)

where:
    accuracy_bonus = 1 if decision == truth else 0  (or soft version)
    calibration_bonus = -Brier_score(probabilities, one_hot(truth))
                       = -(sum_i (p_i - y_i)^2)
    
    (Brier score is a proper scoring rule: it's minimized when p = true probability)
```

**Training loop:**
1. Sample a state + questions from the training distribution
2. Forward pass through the model → structured decisions + probabilities
3. Compute reward = accuracy + calibration
4. Update model parameters to increase expected reward (policy gradient or similar)
5. Repeat

**Key difference from standard RL for LMs:**
- Standard RL for LMs (RLHF, RLVR) operates on **token sequences** — the policy generates tokens one at a time
- RLCD operates on **decision vectors** — the policy produces all decisions in one shot, and the reward is on the calibration of those decisions
- This means the "action space" is the space of probability distributions over valid outputs, not the space of token sequences

### Architecture for Tier 3

If building from scratch (not distilling from Tier 1), the architecture would be:

```
[Custom Encoder] ← designed for decision tasks, not general text
    │
    ├── Text encoder (transformer or SAN-based)
    │   └── Processes state text → hidden states
    │
    ├── Question encoder
    │   └── Embeds question specs → question vectors
    │
    ├── Interaction module
    │   └── State-question attention/pooling
    │
    └── Parallel output heads
        ├── Yes/No head → P(yes), P(no)
        ├── Choice head → P(option_1), ..., P(option_K)
        ├── Score head → value, confidence
        └── Calibration head → calibrated probabilities (replaces raw head outputs)
```

**Architecture choice:**
- If starting from scratch with limited resources: use a small transformer encoder (6–12 layers, 384–768 dim, ~20–50M params)
- If you have the Cactus Platform access: use Needle3's architecture directly
- If you want to experiment with the Simple Attention Network: implement the SAN layers as described in Needle3's documentation

### Training data requirements for Tier 3

RLCD needs **(state, questions, ground_truth_decisions)** triples where:
- The ground truth decision is known (or can be verified programmatically)
- Multiple examples with varying difficulty (so the model learns when to be uncertain)
- Enough volume for RL to converge (100K–10M+ examples depending on complexity)

This is the hardest part — constructing a dataset where you know the "true" calibrated decision. See `DATASETS.md` for construction approaches.

---

## Comparison with Jev

| Architectural decision | Jev (TypeSafe) | This Blueprint |
|-----------------------|----------------|----------------|
| **Text encoding** | Proprietary encoder | Open encoder (MiniLM/BERT/T5) or Needle3 SAN |
| **Question conditioning** | Proprietary | Question spec encoder + attention pooling |
| **Output mechanism** | Parallel decision heads (proprietary) | Parallel MLP heads (standard) |
| **Type safety** | Built into model architecture | Head design (Tier 1) → grammar (Tier 2) |
| **Calibration** | RLCD-trained, calibrated | Temperature scaling (Tier 1) → learned head (Tier 2) → RLCD (Tier 3) |
| **Parallel sampling** | Single forward pass, all outputs | Single forward pass, all heads (same approach) |
| **Training algorithm** | RLCD (proprietary) | Supervised → distillation → RLCD-style (Tier 3) |
| **Training data** | Proprietary 360B+ tokens | Open datasets + synthetic construction |
| **Inference** | Custom engine, 70ms–500ms | PyTorch/HF (Tier 1) → Needle3 engine (Tier 2, faster) |
| **Model size** | Unknown (cloud API) | 22M + heads (Tier 1) → 8–29 MB (Tier 2) |
| **Deployment** | API only | API + edge + mobile + air-gapped (Tier 2) |

### Where we expect to match Jev

- **Structured decision accuracy** on narrow domains (customer support triage, content moderation, intent routing, sentiment scoring) — the encoder has strong text understanding, and the heads are trained on domain-relevant data
- **Type safety** — guaranteed by head design (Tier 1) and grammar (Tier 2), same as Jev
- **Parallel speed** — single forward pass for all questions, same architecture pattern as Jev
- **Calibration** — may not match Jev's RLCD quality initially, but temperature scaling + validation-based calibration gets close for many use cases

### Where we expect to lag Jev

- **General intelligence** — Jev is trained on a much larger and more diverse dataset. Our Tier 1 system is limited by the encoder's general knowledge and the domain-specific training data.
- **Calibration quality** — Jev's RLCD is proprietary and presumably well-engineered. Our temperature scaling is a simpler approach.
- **Workflow evals** — TypeSafe designed their evals around their model. We need to construct comparable eval sets.
- **Speed at scale** — Jev's custom engine is optimized for their architecture. Our Tier 1 runs on general-purpose PyTorch/HF infrastructure (though Tier 2's Needle3 engine should be competitive).

---

## What success looks like

A successful System One model (Tier 1 or 2):

1. **Takes the same input as Jev:** text state + typed question specs
2. **Returns the same shape as Jev:** typed decisions with calibrated probabilities
3. **Matches Jev on decision accuracy** for the domains it's trained on
4. **Has comparable or better calibration** for those domains
5. **Runs in comparable or better time** (especially Tier 2 on edge hardware)
6. **Costs less** (free if self-hosted, vs. Jev's $0.042/MTok)
7. **Is open** — weights, architecture, training methodology all available

---

## Next steps

See the following documents for implementation details:

- [`TRAINING.md`](TRAINING.md) — How to train each tier
- [`DATASETS.md`](DATASETS.md) — What data to use and how to construct it
- [`EVALUATION.md`](EVALUATION.md) — How to measure against Jev
- [`INFRA.md`](INFRA.md) — Infrastructure and deployment
