# Evaluation — Head-to-Head vs Jev

## Overview

How to measure whether your System One model competes with Jev on the dimensions that matter:

1. **Decision accuracy** — does it make the right calls?
2. **Calibration** — do its confidence scores match reality?
3. **Speed** — how fast is it?
4. **Type safety** — does it ever produce invalid output?
5. **Cost** — how much does it cost to run?
6. **Generalization** — does it work on unseen cases?

This document describes an evaluation methodology inspired by TypeSafe's workflow evals, adapted for open benchmarking.

---

## Evaluation philosophy

### What to measure

**Don't measure:** perplexity, token-level accuracy, BLEU/ROUGE, standard LM benchmarks (MMLU, GSM8K, etc.)

These measure text generation quality. System One models don't generate text — they make structured decisions. Measuring them on text generation benchmarks misses the point entirely.

**Do measure:** decision accuracy, calibration quality, inference speed, output validity, cost efficiency.

### How to measure (TypeSafe's approach)

TypeSafe's workflow evals are clever:

1. **Define a workflow** — a compute graph of decisions that produces a final outcome
2. **Run every model through the same workflow** — same input, same questions, same branching logic
3. **Compare to reference probabilities** — use the average of the smartest external models (GPT-6 Astra + Fable 5.1) as "ground truth" probabilities
4. **Measure agreement** — how close are each model's probabilities to the reference?

This avoids the problem of "the harness and model can change together" (overfitting to evals) because the workflow is fixed and the reference is external models, not a constructed ground truth.

### Our adaptation

We can't access GPT-6 Astra or Fable. But we can:

1. Use GPT-4o or Claude as the reference model (accessible via API)
2. Use ensemble of multiple open models as reference
3. Use human-labeled data as ground truth for specific domains
4. Use programmatic verification for domains where correctness is checkable

---

## Evaluation dimensions

### 1. Decision accuracy

**Definition:** How often does the model make the correct decision?

**Measurement:**

For Yes/No questions:
```
Accuracy = (correct predictions) / (total predictions)
```

For Choice questions:
```
Accuracy = (exact match) / (total predictions)
```

For Score questions:
```
Accuracy = within-1-level accuracy = (predictions within 1 level of truth) / (total)
```
Or use ordinal correlation (Spearman's rho between predicted and true scores).

**Granularity:** Measure per question type, per domain, per difficulty level.

**Difficulty stratification:**
- Easy: high consensus among labelers / reference models
- Medium: moderate disagreement
- Hard: low consensus, genuinely ambiguous

Report accuracy separately for each difficulty level. A model that's 95% accurate on easy cases but 50% on hard cases is different from one that's 80% on both.

### 2. Calibration quality

**Definition:** When the model says "90% confident," is it right 90% of the time?

**Primary metric: Expected Calibration Error (ECE)**

```
ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

where:
  B_b = set of predictions in confidence bin b
  acc(B_b) = accuracy in bin b
  conf(B_b) = average confidence in bin b
  N = total predictions
```

Lower is better. ECE < 0.05 is good. ECE > 0.15 is poor.

**Secondary metrics:**

**Brier Score:**
```
Brier = (1/N) * sum_i sum_k (p_ik - y_ik)^2

where:
  p_ik = predicted probability for option k on example i
  y_ik = 1 if k is the true answer for example i, else 0
```

Lower is better. This is a proper scoring rule — it incentivizes honest probabilities.

**Reliability diagram:**
Plot accuracy vs. confidence across bins. A perfectly calibrated model follows the diagonal.

**Overconfidence/underconfidence:**
Measure the gap between average confidence and average accuracy. Positive gap = overconfidence (common for neural networks). Negative gap = underconfidence.

### 3. Inference speed

**Definition:** End-to-end time from input to output.

**Measurement:**
```
Latency = time_from_request_to_response
```

Measure:
- **Per-call latency:** time for one (state + questions) → decisions call
- **P50, P90, P99:** latency distribution (tail latency matters for production)
- **Throughput:** calls per second at concurrency N

**Breakdown (if possible):**
- Time to encode state
- Time for forward pass
- Time for head computation
- Time for output construction / grammar decoding

**Hardware reporting:** Always report hardware (CPU model, GPU model, or "cloud API").

### 4. Type safety

**Definition:** Does the model produce valid output that matches the schema?

**Measurement:**
```
Type error rate = (outputs that fail schema validation) / (total outputs)
```

For System One models, this should be **zero** by construction:
- Tier 1: heads output valid probability distributions → output construction guarantees valid JSON
- Tier 2: grammar-constrained decode guarantees valid output

This is a pass/fail metric. If the model ever produces an invalid output, it fails.

### 5. Cost

**Definition:** Cost per decision (or per 1000 decisions).

**For self-hosted (Tier 1/2):**
```
Cost = compute_cost + storage_cost + maintenance_cost

compute_cost = (GPU_hours * GPU_hourly_rate) + (CPU_hours * CPU_hourly_rate)
```

For small models on CPU, this can be near-zero. For GPU inference, depends on GPU pricing.

**For API-based (Jev comparison point):**
```
Jev cost = $0.042 per million input tokens
         = $0.042 / (input_tokens / 1_000_000)
```

For a typical decision query with 500 input tokens:
```
Jev cost per query = $0.042 * 500 / 1_000_000 = $0.000021 = 0.0021 cents
```

**Comparison:** If your self-hosted model runs on a $5/month VPS and handles 10K queries/day, cost per query is $5 / (30 * 10000) = $0.000017 = 0.0017 cents — competitive with Jev on cost, and free if you already have the hardware.

### 6. Generalization

**Definition:** How well does the model work on data from a different distribution than training?

**Measurement:**
- Train on domain A, test on domain B
- Train on time period T1, test on T2
- Train on source S1, test on source S2

**Domain shift types:**
- **Topic shift:** training on news, testing on support tickets
- **Language shift:** training on English, testing on other languages (Jev supports English primarily)
- **Style shift:** training on formal text, testing on informal/social media
- **Length shift:** training on short text, testing on long documents

**Report:** accuracy drop from in-domain to out-of-domain. Smaller drop = better generalization.

---

## Evaluation harness design

### Workflow evaluation (TypeSafe-style)

Define workflows as code. Each workflow is a compute graph of decisions:

```python
class SupportTicketWorkflow:
    """
    Workflow: classify a support ticket and decide what to do.
    """
    def __init__(self, model):
        self.model = model
    
    def run(self, ticket_text):
        # Step 1: Classify the ticket
        classification = self.model.classify(
            state=ticket_text,
            questions=[
                Question("team", "choice", ["billing", "technical", "sales", "account"]),
                Question("urgency", "yes_no", "Does this need response today?"),
                Question("sentiment", "score", ["calm", "mildly_annoyed", "frustrated", "angry_churn"]),
                Question("churn_risk", "yes_no", "Is the customer likely to churn?"),
            ]
        )
        
        # Step 2: Branching logic (domain-specific)
        if classification["churn_risk"]["answer"] == "yes" and classification["urgency"]["answer"] == "yes":
            action = "escalate_to_manager"
        elif classification["team"]["answer"] == "billing":
            action = "route_to_billing"
        else:
            action = "route_to_general"
        
        return {
            "classification": classification,
            "action": action,
        }
```

Then evaluate:

```python
def evaluate_workflow(workflow, test_data, reference_model=None):
    results = []
    
    for example in test_data:
        # Run through the workflow
        output = workflow.run(example["ticket_text"])
        
        # Compare to reference (if available)
        if reference_model:
            reference_output = reference_model.run(example["ticket_text"])
            agreement = compare_outputs(output, reference_output)
        else:
            agreement = compare_to_ground_truth(output, example["expected"])
        
        results.append({
            "example_id": example["id"],
            "output": output,
            "agreement": agreement,
            "latency": output["latency"],
        })
    
    return aggregate_results(results)
```

### Standard benchmark evaluation

In addition to workflow evals, measure on standard benchmarks converted to decision format:

| Benchmark | Original task | System One conversion |
|-----------|--------------|----------------------|
| **SNIPS** | Slot filling | Field-level extraction → Score/Choice questions |
| **DSTC8** | Dialogue state tracking | Dialogue state → structured decision |
| **SST-2 / SST-5** | Sentiment analysis | Text → sentiment Choice/Score |
| **BoolQ** | Yes/No QA | Passage + question → Yes/No decision |
| **TREC QA** | Question classification | Question → question type Choice |
| **AG News** | News classification | Text → topic Choice |

Convert each benchmark to System One format and measure accuracy + calibration.

---

## Head-to-head comparison methodology

### Setup

To compare your model against Jev:

1. **Select a common set of workflows** — same workflows for both models
2. **Select a common set of test examples** — same inputs for both models
3. **Run both models** — get decisions + confidence from each
4. **Compare across all dimensions**

### What you need to compare against Jev

**Access to Jev:** You need Jev API access (early access from TypeSafe, or use jev-ai.pro as a proxy).

**Common input format:** Jev accepts (state, model, questions) JSON. Your model should accept the same format for fair comparison.

**Common output format:** Jev returns (answer, probabilities, confidence) per question. Your model should return the same shape.

### Comparison dimensions

| Dimension | How to compare |
|-----------|---------------|
| **Accuracy** | Same examples, same questions, compare decisions to reference |
| **Calibration** | Same examples, compare ECE and Brier scores |
| **Speed** | Same hardware conditions (or note the difference) |
| **Type safety** | Pass/fail — does either model produce invalid output? |
| **Cost** | Compute your cost vs. Jev's published pricing |
| **Output quality** | Qualitative review of disagreement cases |

### Disagreement analysis

When your model and Jev disagree, analyze:

1. **Who's right?** — check against reference / ground truth
2. **Confidence comparison** — who was more confident? Was the confidence justified?
3. **Pattern analysis** — do disagreements cluster in specific domains, question types, or input patterns?
4. **Failure mode analysis** — what kinds of inputs cause each model to fail?

This analysis is often more valuable than aggregate metrics because it tells you *where* each model is strong and weak.

---

## Evaluation dataset construction

### Minimum viable eval set

| Component | Size | Purpose |
|-----------|------|---------|
| **In-domain test** | 500–2000 examples | Measure accuracy on intended domain |
| **Out-of-domain test** | 200–500 examples | Measure generalization |
| **Calibration set** | 500–1000 examples | Measure calibration quality |
| **Workflow set** | 10–50 workflows, 10–100 examples each | Measure end-to-end workflow performance |
| **Adversarial set** | 100–500 examples | Measure robustness to edge cases, adversarial inputs |

### Reference model options

For "ground truth" decisions:

1. **GPT-4o / Claude 3.5** — accessible, strong, but may have biases
2. **Ensemble of open models** — Llama-3-70B, Qwen-2.5-72B, etc. — less biased but may be weaker
3. **Human labelers** — most reliable but expensive and slow
4. **Programmatic verification** — only available for specific domains (math, code, format validation)

**Recommendation:** Use GPT-4o as the primary reference (same approach as TypeSafe but with an accessible model), supplemented by human labels on a subset for calibration.

### Eval data sources

1. **Convert open benchmarks** (SNIPS, SST, BoolQ, etc.) → System One format
2. **Collect domain-specific examples** (your actual use case data)
3. **Generate synthetic examples** (oracle-labeled, rule-based)
4. **Hand-label a gold set** (most valuable for calibration evaluation)

---

## Reporting results

### Standard result table

| Model | Accuracy (overall) | ECE | Brier | Latency (P50) | Latency (P99) | Type errors | Cost/query |
|-------|-------------------|-----|-------|---------------|---------------|-------------|------------|
| Jev (TypeSafe) | X% | X | X | Xms | Xms | 0% | $0.000021 |
| System One Tier 1 | X% | X | X | Xms | Xms | 0% | $0.0000X |
| System One Tier 2 | X% | X | X | Xms | Xms | 0% | $0.0000X |

### Per-question-type breakdown

| Model | Yes/No Acc | Choice Acc | Score Acc (within-1) | Yes/No ECE | Choice ECE | Score ECE |
|-------|-----------|------------|---------------------|------------|------------|-----------|
| Jev | X% | X% | X% | X | X | X |
| Our model | X% | X% | X% | X | X | X |

### Per-domain breakdown (if multiple domains)

| Model | Domain A Acc | Domain B Acc | Domain C Acc |
|-------|-------------|-------------|-------------|
| Jev | X% | X% | X% |
| Our model | X% | X% | X% |

### Disagreement analysis

| Category | Count | Our model right | Jev right | Both wrong | Tie |
|----------|-------|----------------|-----------|------------|-----|
| Overall | N | X% | X% | X% | X% |
| By difficulty: Easy | N | X% | X% | X% | X% |
| By difficulty: Medium | N | X% | X% | X% | X% |
| By difficulty: Hard | N | X% | X% | X% | X% |
| By question type: Yes/No | N | X% | X% | X% | X% |
| By question type: Choice | N | X% | X% | X% | X% |
| By domain: A | N | X% | X% | X% | X% |

---

## Success criteria

### Minimum bar (Tier 1 must hit this to be worth pursuing Tier 2)

- [ ] **Accuracy within 10% of Jev** on in-domain test set (same questions, same examples)
- [ ] **ECE < 0.15** (calibration is usable — not great but not broken)
- [ ] **Zero type errors** (by construction, but verify)
- [ ] **Latency under 500ms** per query (matches Jev's upper bound)
- [ ] **Cost competitive with Jev** (self-hosted cost per query < $0.0001)

### Tier 2 target (distilled Needle3)

- [ ] **Accuracy within 5% of Jev** on in-domain test set
- [ ] **ECE < 0.10** (good calibration)
- [ ] **Zero type errors** (guaranteed by grammar)
- [ ] **Latency under 100ms** on CPU (edge deployment viable)
- [ ] **Model size under 30 MB** (quantized .cact)
- [ ] **Runs on Raspberry Pi 5** (or equivalent edge hardware)

### Tier 3 target (full RLCD)

- [ ] **Accuracy matches or exceeds Jev** on in-domain test set
- [ ] **ECE < 0.05** (excellent calibration, matches or exceeds Jev)
- [ ] **Zero type errors**
- [ ] **Latency competitive with Jev** (70ms–500ms)
- [ ] **Better calibration than Jev** on at least some domains (the RLCD advantage)
- [ ] **Generalizes better than Jev** on out-of-domain tests (if training data is more diverse)

---

## What you can evaluate right now (without building the model)

Before building anything, you can run a **proxy evaluation** using existing models as stand-ins for your future System One model:

1. **Pick a reference workflow** (e.g., support ticket triage with 4 questions)
2. **Collect 100–500 test examples** (real or synthetic)
3. **Run each example through:**
   - Jev (via API, if you have access)
   - GPT-4o with structured output (constrained to Yes/No/Choice/Score format)
   - GPT-4o with chain-of-thought (generate reasoning, then decide)
   - A small open model (MiniLM + simple classifier heads, as a Tier 1 stand-in)
4. **Compare all of them** on accuracy, calibration, and speed
5. **Identify the gap** between the best available proxy and Jev

This tells you:
- What's achievable with current open components (the Tier 1 ceiling)
- What Jev actually does better (the gap to close)
- Whether the gap is worth closing (is Jev's advantage significant for your use case?)

---

## Summary

The evaluation methodology is:

1. **Define workflows** that represent real use cases
2. **Collect diverse test data** with ground truth (human, reference model, or programmatic)
3. **Run all models through the same workflows** with the same inputs
4. **Measure accuracy, calibration, speed, type safety, and cost**
5. **Analyze disagreements** to understand where each model is strong and weak
6. **Report transparently** — show the breakdowns, not just aggregate numbers

The goal is not to "beat Jev" on every metric. The goal is to build a model that is **competitive** on the dimensions that matter for your use case, while being **open, self-hostable, and deployable on edge hardware**.

---

## Reference

TypeSafe's evaluation approach: https://typesafe.ai/blog/introducing-system-one-models-and-jev (see "Workflow evals" section)

Key metrics:
- **ECE** (Expected Calibration Error): standard metric for calibration quality
- **Brier Score**: proper scoring rule for probabilistic predictions
- **Reliability diagrams**: visual calibration assessment
- **Workflow evals**: TypeSafe's novel evaluation methodology
