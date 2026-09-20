# Datasets — System One Model

## Overview

Training a System One model requires **(state, questions, decisions)** triples where:
- **state** = unstructured text (customer message, support ticket, document excerpt, etc.)
- **questions** = typed decision questions (Yes/No, Choice, Score)
- **decisions** = ground truth answers with confidence/probability distributions

This document covers:
1. Open datasets available on HuggingFace
2. How to convert existing datasets into System One format
3. Synthetic data construction using larger models
4. Hand-labeled data strategy

---

## Open dataset inventory

### A. Function calling / tool use → Decision data

These datasets have (text, tool_schema, tool_call) triples — directly mappable to (state, questions, decisions).

| Dataset | Downloads | Format | Mapping to System One |
|---------|-----------|--------|----------------------|
| **Salesforce/xlam-function-calling-60k** | 35,888 | (conversation, tools, call) | Tool selection → Choice question. Arguments → structured extraction. |
| **NousResearch/hermes-function-calling-v1** | 61,266 | (prompt, tools, call) | Same as above. Large volume. |
| **glaiveai/glaive-function-calling-v2** | 53,677 | (prompt, tools, call) | Same pattern. |
| **stindardlogic/tool-calling-english-100k** | 369 | Tool calls in English | Direct mapping. |
| **nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1** | 1,914 | RL-focused function calling | Higher quality, smaller volume. |

**Conversion approach:**
```python
# Function calling example → System One example
fc_example = {
    "conversation": "What's the weather in Lagos?",
    "tools": [
        {"name": "get_weather", "parameters": {"city": "string"}},
        {"name": "get_time", "parameters": {"timezone": "string"}},
    ],
    "call": {"name": "get_weather", "arguments": {"city": "Lagos"}}
}

# → System One format
system_one_example = {
    "state": "What's the weather in Lagos?",
    "questions": [
        {
            "id": "which_tool",
            "type": "choice",
            "options": ["get_weather", "get_time"],
            "description": "Which tool should be called?",
        }
    ],
    "answers": {
        "which_tool": {
            "choice": "get_weather",
            "probabilities": {"get_weather": 0.95, "get_time": 0.05},
            "confidence": 0.95,
        }
    }
}
```

**Volume potential:** 60K–150K examples from function-calling datasets alone.

### B. Classification / sentiment → Score and Yes/No data

| Dataset | Downloads | Format | Mapping |
|---------|-----------|--------|---------|
| **Karavet/ILUR-news-text-classification-corpus** | 3,925 | News text → category | Text → Choice question (category selection) |
| **Husain/intent-classification-en-fr** | 124 | Text → intent | Text → Choice question (intent selection) |
| **GuardrailsAI/content-moderation** | 53 | Text → moderation decision | Text → Yes/No (is this harmful?) + Score (severity) |
| **jakeazcona/short-text-labeled-emotion-classification** | 350 | Text → emotion | Text → Choice (emotion) + Score (intensity) |

**Conversion approach for sentiment → Score:**
```python
sentiment_example = {
    "text": "I love this product, it's amazing!",
    "label": "positive",
    "score": 0.95,  # if available
}

# → System One format
system_one_example = {
    "state": "I love this product, it's amazing!",
    "questions": [
        {
            "id": "sentiment",
            "type": "choice",
            "options": ["positive", "neutral", "negative"],
            "description": "What is the sentiment?",
        },
        {
            "id": "sentiment_intensity",
            "type": "score",
            "scale": ["very_low", "low", "medium", "high", "very_high"],
            "description": "How intense is the sentiment?",
        }
    ],
    "answers": {
        "sentiment": {
            "choice": "positive",
            "probabilities": {"positive": 0.95, "neutral": 0.04, "negative": 0.01},
            "confidence": 0.95,
        },
        "sentiment_intensity": {
            "value": 4.0,
            "answer": "very_high",
            "probabilities": {"very_low": 0.01, "low": 0.02, "medium": 0.07, "high": 0.30, "very_high": 0.60},
            "confidence": 0.60,
        }
    }
}
```

**Volume potential:** Tens of thousands of classification examples across available datasets.

### C. Structured extraction → Choice + Yes/No data

| Dataset | Downloads | Format | Mapping |
|---------|-----------|--------|---------|
| **stindardlogic/structured-output-sft-100k** | 75 | Text → structured JSON | Text → multiple extraction questions |
| **Arun63/sharegpt-structured-output-json** | 1,481 | Conversation → JSON | Multi-field extraction |
| **paraloq/json_data_extraction** | 221 | Text → JSON fields | Field-level extraction → individual questions |

**Conversion approach:**
A structured extraction task naturally decomposes into multiple independent decisions:
```python
extraction_example = {
    "text": "Invoice from Acme Corp, $1,200.00, due 2026-09-01",
    "schema": {"vendor": "str", "total": "float", "due_date": "str"},
    "result": {"vendor": "Acme Corp", "total": 1200.0, "due_date": "2026-09-01"}
}

# → System One format: each field is a question
system_one_example = {
    "state": "Invoice from Acme Corp, $1,200.00, due 2026-09-01",
    "questions": [
        {"id": "vendor", "type": "choice", "options": [...], "description": "What is the vendor?"},
        {"id": "total", "type": "score", "scale": [...], "description": "What is the total amount?"},
        {"id": "due_date", "type": "choice", "options": [...], "description": "What is the due date?"},
    ],
    "answers": {
        "vendor": {"choice": "Acme Corp", "confidence": 0.98, ...},
        "total": {"value": 1200.0, "confidence": 0.95, ...},
        "due_date": {"choice": "2026-09-01", "confidence": 0.97, ...},
    }
}
```

### D. Calibration-specific datasets

| Dataset | Downloads | Use |
|---------|-----------|-----|
| **TaylorAI/RLCD-generated-preference-data** | 76 | Directly relevant — RLCD preference data (small) |
| **Anthropic/hh-rlhf** | 37,941 | Preference pairs — useful for reward modeling, though RLHF-focused |
| **myyycroft/granular-uncertainty-quantification-dataset** | 16 | Uncertainty quantification — directly relevant but small |

These are smaller but directly relevant to the calibration aspect of System One models.

---

## Synthetic data construction

Large models (GPT-4, Claude, Gemini) can generate high-quality labeled data for System One training. This is how TypeSafe constructed some of their eval data.

### Approach 1: Oracle labeling

Use a strong model to label data with calibrated probabilities.

```python
import openai

def oracle_label(state, questions, model="gpt-4"):
    """
    Use a strong LLM to produce calibrated labels.
    Prompt the model to output probabilities, not just decisions.
    """
    prompt = f"""
    Analyze the following text and answer each question with a probability distribution.
    
    Text: {state}
    
    Questions:
    """
    for q in questions:
        prompt += f"\n{q['description']}\nType: {q['type']}\n"
        if q['type'] == 'choice':
            prompt += f"Options: {q['options']}\n"
        elif q['type'] == 'score':
            prompt += f"Scale: {q['scale']}\n"
    
    prompt += """
    
    For each question, provide:
    1. Your best answer
    2. A probability distribution over all possible answers
    3. Your confidence in your best answer (0–1)
    
    Important: Be honest about uncertainty. If the answer is ambiguous, assign meaningful probability to multiple options.
    """
    
    response = openai.chat.complete(model=model, messages=[{"role": "user", "content": prompt}])
    return parse_oracle_response(response)
```

**Quality considerations:**
- The oracle model's calibration becomes your training target. If GPT-4 is overconfident, your model learns to be overconfident.
- Use the oracle's probabilities directly as training targets, not just the argmax decision.
- For critical use cases, verify oracle labels against human judgment on a sample.

**Volume:** Can generate 10K–100K+ examples depending on budget and rate limits.

### Approach 2: Data augmentation

Take existing labeled data and create variations:

```python
def augment_example(example, num_variations=5):
    """
    Create variations of a training example:
    - Paraphrase the state text
    - Rephrase the questions
    - Add noise/perturbations
    """
    variations = []
    for i in range(num_variations):
        variant = {
            "state": paraphrase(example["state"]),
            "questions": rephrase_questions(example["questions"]),
            "answers": example["answers"],  # same answers (content preserved)
        }
        variations.append(variant)
    return variations
```

**Tools:** Use a paraphrase model (e.g., T5-based paraphrasing) or a larger LLM to generate variations.

### Approach 3: Programmatic generation for verifiable domains

For domains where correctness can be checked programmatically, generate data with known ground truth.

```python
def generate_rule_based_data(rules, num_examples=10000):
    """
    Generate examples where the correct decision is determined by rules.
    E.g., for content moderation: generate text with/without specific patterns.
    """
    data = []
    for _ in range(num_examples):
        # Generate text that matches or doesn't match rules
        text, true_label = generate_from_rules(rules)
        
        data.append({
            "state": text,
            "questions": [{"id": "is_toxic", "type": "yes_no", "description": "Is this content toxic?"}],
            "answers": {
                "is_toxic": {
                    "yes_no": "yes" if true_label else "no",
                    "confidence": 1.0,  # rule-based = certain
                }
            }
        })
    return data
```

**Use cases:** Content moderation (keyword/pattern-based), intent routing (keyword-based), format validation (schema-based), math/logic decisions (computation-based).

**Advantage:** Ground truth is known with certainty → perfect calibration targets.

**Limitation:** Rules are brittle and may not capture the full complexity of real-world decisions.

---

## Hand-labeled data strategy

For your specific use case, hand-labeled data is the most valuable because it reflects your actual decision criteria.

### What to label

1. **Seed set (500–2000 examples):** Diverse examples covering all question types, difficulty levels, and edge cases. This is your gold standard.

2. **Calibration set (500–1000 examples):** Separate from train/test. Used specifically to learn the calibration layer.

3. **Hard cases (200–500 examples):** Examples where the decision is genuinely ambiguous. These teach the model when to be uncertain.

### How to label

For each example, labelers should provide:
1. The best answer for each question
2. A probability distribution (how likely is each option?)
3. A confidence score (how sure are you?)

**Labeling interface sketch:**
```
State: "Payouts have failed since Monday. Fix it today or we switch provider."

Question 1: Which team should handle this?
  [ ] billing    [ ] technical    [ ] sales    [ ] account
  Probabilities: billing ___%  technical ___%  sales ___%  account ___%
  Confidence: ___%

Question 2: Does the customer need a response today?
  [ ] Yes  [ ] No
  Probability yes: ___%
  Confidence: ___%

Question 3: How frustrated is the customer?
  [ ] Calm  [ ] Mildly annoyed  [ ] Frustrated  [ ] Angry and ready to churn
  Probabilities: calm ___%  mildly ___%  frustrated ___%  angry ___%
  Confidence: ___%
```

### Labeling quality

- **Multiple labelers per example:** 2–3 labelers, measure inter-annotator agreement
- **Calibration check:** Ask labelers to provide probabilities, then check if their confidence matches their accuracy
- **Disagreement handling:** When labelers disagree, the example is genuinely ambiguous → the model should learn low confidence on similar cases

---

## Dataset construction recipes

### Recipe 1: Customer support triage (use case example)

**Goal:** Classify support tickets by team, urgency, sentiment, churn risk.

**Data sources:**
1. Open intent classification datasets (intent → team mapping)
2. Open sentiment analysis datasets (text → sentiment score)
3. Synthetic data from oracle (generate realistic support tickets)
4. Hand-labeled internal data (actual tickets with team assignments)

**Question set:**
- `team` (Choice): billing, technical, sales, account, product
- `urgency` (Yes/No): does this need response today?
- `sentiment` (Score): calm → angry_churn
- `churn_risk` (Yes/No): is this customer likely to leave?

**Estimated data volume:**
- Open datasets: ~50K examples (converted)
- Synthetic: ~50K examples (oracle-labeled)
- Hand-labeled: ~1K examples (gold standard)
- **Total: ~100K examples**

### Recipe 2: Content moderation

**Goal:** Classify content as safe/toxic, severity, category.

**Data sources:**
1. GuardrailsAI/content-moderation (53 examples)
2. Toxicity datasets on HF (search "toxicity", "hate speech", "moderation")
3. Oracle-labeled diverse content
4. Rule-based generation (known toxic/safe patterns)

**Question set:**
- `is_toxic` (Yes/No)
- `severity` (Score): safe → severely_toxic
- `category` (Choice): hate_speech, harassment, sexual, violence, self_harm, other

### Recipe 3: Text classification benchmark conversion

Take any text classification dataset and convert to System One format:

```python
def convert_classification_to_system_one(dataset, label_names):
    """
    Convert a standard text classification dataset to System One format.
    
    dataset: HuggingFace dataset with 'text' and 'label' columns
    label_names: list of label names corresponding to integer labels
    """
    system_one_data = []
    
    for example in dataset:
        system_one_data.append({
            "state": example["text"],
            "questions": [
                {
                    "id": "classification",
                    "type": "choice",
                    "options": label_names,
                    "description": "Classify this text",
                }
            ],
            "answers": {
                "classification": {
                    "choice": label_names[example["label"]],
                    "probabilities": one_hot(example["label"], len(label_names)),
                    "confidence": 1.0,  # ground truth = certain
                }
            }
        })
    
    return system_one_data
```

---

## Data volume guidelines

| Model tier | Minimum data | Good data | Excellent data |
|------------|-------------|-----------|----------------|
| **Tier 1 (frozen encoder + heads)** | 1K–5K examples | 10K–50K examples | 50K–200K examples |
| **Tier 2 (distillation into Needle3)** | 10K examples (from Tier 1) | 100K examples | 1M+ examples |
| **Tier 3 (RLCD from scratch)** | 50K examples | 500K examples | 5M+ examples |

**More data helps most when:**
- The decision domain is complex (many edge cases)
- The questions are nuanced (ambiguous cases are common)
- You want good calibration (need examples of ambiguous cases to learn uncertainty)

**Less data is needed when:**
- The domain is narrow and well-defined
- The decisions are mostly clear-cut
- You're using a strong pretrained encoder (the encoder brings significant knowledge)

---

## Data quality checklist

Before training, verify:

- [ ] Each example has valid (state, questions, answers) structure
- [ ] Answer probabilities sum to 1.0 (within tolerance)
- [ ] Confidence scores are in [0, 1]
- [ ] No data leakage between train/val/test/calibration sets
- [ ] Category distribution is reasonable (not all examples are "easy")
- [ ] Ambiguous cases are present (for calibration learning)
- [ ] Domain coverage matches intended deployment (don't train on news and deploy on support tickets)
- [ ] Label quality is verified (spot-check 50–100 examples)

---

## Dataset storage format

Use a simple JSONL format for training data:

```jsonl
{"state": "...", "questions": [...], "answers": {...}}
{"state": "...", "questions": [...], "answers": {...}}
...
```

This is:
- Human-readable
- Streamable (can read line by line for large datasets)
- Compatible with HuggingFace datasets library
- Easy to generate and manipulate

---

## HuggingFace dataset publishing

If you create a high-quality System One dataset, consider publishing it on HuggingFace:

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload dataset
huggingface-cli upload your-username/system-one-decision-dataset \
    data/*.jsonl \
    --repo-type dataset
```

This contributes back to the open ecosystem and lets others build on your work.

---

## Summary

| Data source | Volume potential | Quality | Effort |
|-------------|-----------------|---------|--------|
| Open classification datasets | 10K–100K | Medium (depends on dataset) | Low (automated conversion) |
| Open function-calling datasets | 60K–150K | Medium-High | Low (direct mapping) |
| Oracle labeling (GPT-4/Claude) | 10K–100K+ | Medium-High (depends on oracle) | Medium (prompt design + parsing) |
| Hand-labeled | 500–5000 | High (your domain) | High (manual effort) |
| Rule-based generation | 10K–1M+ | High (for rule-covered cases) | Medium (rule design) |
| Data augmentation | 2–10x existing data | Medium (variations of existing) | Low (automated) |
