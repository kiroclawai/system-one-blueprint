# Training Pipeline — System One Model

## Overview

Three-tier training pipeline, from quickest (Tier 1) to most ambitious (Tier 3).

| Tier | Approach | Duration (1 GPU) | Data needed | Expected outcome |
|------|----------|-----------------|-------------|-----------------|
| **Tier 1** | Supervised training of decision heads on frozen/open encoder | Hours–days | 10K–100K labeled examples | Working Jev-compatible decision system |
| **Tier 2** | Distill Tier 1 into Needle3 + calibration fine-tuning | Days–weeks | 100K–10M labeled examples (can be synthetic from Tier 1) | 8–29 MB deployable model with grammar-constrained output |
| **Tier 3** | RLCD-style end-to-end training | Weeks–months | 100K–10M+ verifiable (state, questions, truth) triples | Custom model trained for calibrated decisions from scratch |

---

## Tier 1: Supervised Head Training

### Goal

Train decision heads on top of a pretrained encoder to produce calibrated Yes/No, Choice, and Score outputs from text state + question specs.

### Phase 1a: Data preparation

**Input format:**
```python
# Each training example
{
    "state": "Customer: 'Payouts have failed since Monday. Fix it today or we switch provider.'",
    "questions": [
        {"id": "team", "type": "choice", "options": ["billing", "technical", "sales", "account"]},
        {"id": "needs_response_today", "type": "yes_no"},
        {"id": "frustration", "type": "score", "scale": ["calm", "mildly_annoyed", "frustrated", "angry_churn"]},
        {"id": "threatens_churn", "type": "yes_no"},
    ],
    "answers": {
        "team": {"choice": "billing", "confidence": 0.96},
        "needs_response_today": {"yes_no": "yes", "confidence": 0.96},
        "frustration": {"score": "angry_churn", "value": 3.0, "confidence": 0.95},
        "threatens_churn": {"yes_no": "yes", "confidence": 0.98},
    }
}
```

**Constructing training data from open sources:**

See [`DATASETS.md`](DATASETS.md) for the full inventory. Key approaches:

1. **Convert existing classification datasets:** Take a sentiment analysis dataset → frame as Score question. Take an intent classification dataset → frame as Choice question. Take a binary classification dataset → frame as Yes/No question.

2. **Convert function-calling datasets:** Tool calls are decisions — "which tool to call" is a Choice question, "should we call a tool" is a Yes/No question.

3. **Synthetic data from a larger model:** Use GPT-4/Claude to label data with probabilities. This is how TypeSafe constructed some of their eval data — use a strong model as an "oracle" to generate calibrated labels.

4. **Hand-labeled seed set:** For your specific use case (customer support triage, content moderation, etc.), hand-label 500–5000 examples. This is the most valuable data because it reflects your actual decision criteria.

**Data splitting:**
- Train: 80%
- Validation (for calibration): 10%
- Test (for final eval): 10%

**Calibration set:** Keep a separate held-out set specifically for learning the calibration layer (temperature scaling). This should not be the same as the test set.

### Phase 1b: Model setup

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SystemOneModel(nn.Module):
    def __init__(self, encoder_name="sentence-transformers/all-MiniLM-L6-v2", 
                 num_questions=10, hidden_dim=384):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
        # Question embedding table (type + options encoded)
        self.question_encoder = QuestionEncoder(hidden_dim)
        
        # Pooling: CLS token + question embedding → per-question representation
        self.pooler = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Heads: one per question (but all share the same architecture per type)
        self.yes_no_heads = nn.ModuleDict()
        self.choice_heads = nn.ModuleDict()
        self.score_heads = nn.ModuleDict()
        
        # Calibration: temperature per head
        self.temperatures = nn.ParameterDict()
    
    def forward(self, state_texts, question_specs):
        # Encode state
        encoded = self.encoder(**self.tokenizer(state_texts, padding=True, 
                                                truncation=True, return_tensors="pt"))
        cls_repr = encoded.last_hidden_state[:, 0, :]  # [batch, dim]
        
        outputs = {}
        for q_spec, q_repr in zip(question_specs, self.question_encoder(question_specs)):
            combined = torch.cat([cls_repr, q_repr], dim=-1)
            pooled = self.pooler(combined)
            
            if q_spec["type"] == "yes_no":
                head = self.yes_no_heads[q_spec["id"]]
                logits = head(pooled)
                probs = torch.softmax(logits / self.temperatures[q_spec["id"]], dim=-1)
                outputs[q_spec["id"]] = {
                    "probabilities": probs,
                    "confidence": probs.max(dim=-1).values,
                    "logits": logits,
                }
            elif q_spec["type"] == "choice":
                head = self.choice_heads[q_spec["id"]]
                logits = head(pooled)
                probs = torch.softmax(logits / self.temperatures[q_spec["id"]], dim=-1)
                outputs[q_spec["id"]] = {
                    "probabilities": probs,
                    "confidence": probs.max(dim=-1).values,
                    "logits": logits,
                }
            elif q_spec["type"] == "score":
                head = self.score_heads[q_spec["id"]]
                logits = head(pooled)
                probs = torch.softmax(logits / self.temperatures[q_spec["id"]], dim=-1)
                level_indices = torch.arange(head.out_features, device=logits.device).float()
                expected = (probs * level_indices).sum(dim=-1)
                outputs[q_spec["id"]] = {
                    "probabilities": probs,
                    "confidence": probs.max(dim=-1).values,
                    "expected_value": expected,
                    "logits": logits,
                }
        
        return outputs
```

### Phase 1c: Loss function

**Multi-head loss:**

```python
def training_loss(model_outputs, ground_truth, temperatures):
    """
    Combined loss across all question types.
    """
    total_loss = 0.0
    losses = {}
    
    for q_id, output in model_outputs.items():
        gt = ground_truth[q_id]
        
        if output["probabilities"].shape[-1] == 2:  # Yes/No
            # Convert "yes"/"no" to index
            target_idx = 1 if gt["yes_no"] == "yes" else 0
            target = torch.tensor([target_idx], dtype=torch.long)
            loss = nn.CrossEntropyLoss()(output["logits"], target)
            
        elif len(output["probabilities"].shape) == 2:  # Choice or Score
            # Convert answer to index
            if "choice" in gt:
                target_idx = ...  # index of chosen option
            else:  # score
                target_idx = int(gt["value"])  # or soft target from probabilities
            target = torch.tensor([target_idx], dtype=torch.long)
            loss = nn.CrossEntropyLoss()(output["logits"], target)
        
        # Calibration bonus: penalize overconfidence on wrong answers
        # This is a soft version of what RLCD does
        correct = (output["probabilities"].argmax(dim=-1) == target).float()
        confidence = output["confidence"]
        calibration_penalty = torch.abs(confidence - correct)
        
        total_loss += loss + 0.1 * calibration_penalty
        losses[q_id] = loss.item()
    
    return total_loss, losses
```

**Calibration-aware loss (optional,Tier 1+):**

The calibration penalty above is a simple proxy. Better approaches:

1. **Brier score loss** (proper scoring rule):
```python
def brier_score(probabilities, true_one_hot):
    """
    Brier score: mean squared error between predicted probabilities and true outcomes.
    Lower is better. Proper scoring rule → incentivizes honest probabilities.
    """
    return ((probabilities - true_one_hot) ** 2).sum(dim=-1).mean()
```

2. **Expected Calibration Error (ECE) penalty:**
Bin predictions by confidence, measure accuracy in each bin, penalize divergence.

3. **Negative log-likelihood + calibration regularizer:**
```python
nll = nn.NLLLoss()(torch.log(probs + 1e-10), target)
cal_reg = ece(probs, target)  # or brier_score
loss = nll + lambda_cal * cal_reg
```

### Phase 1d: Training loop

```python
def train_tier1(model, train_data, val_data, epochs=10, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_data:
            optimizer.zero_grad()
            outputs = model(batch["states"], batch["questions"])
            loss, _ = training_loss(outputs, batch["answers"], model.temperatures)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation + calibration
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_data:
                outputs = model(batch["states"], batch["questions"])
                loss, _ = training_loss(outputs, batch["answers"], model.temperatures)
                val_loss += loss.item()
                
                # Collect for calibration evaluation
                for q_id, out in outputs.items():
                    all_probs.append(out["probabilities"])
                    # ... collect targets
        
        # Learn temperature scaling on validation set
        learn_calibration(model, val_data)
        
        # Evaluate calibration
        ece = compute_ece(all_probs, all_targets)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, ECE={ece:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
        
        scheduler.step()
```

### Phase 1e: Calibration (temperature scaling)

```python
def learn_calibration(model, cal_data):
    """
    Learn temperature per head on calibration set.
    Optimizes NLL (or Brier score) on held-out data.
    """
    temps = {}
    
    for q_id in model.temperatures.keys():
        # Collect logits and targets for this question
        logits_list = []
        targets_list = []
        
        model.eval()
        with torch.no_grad():
            for batch in cal_data:
                outputs = model(batch["states"], batch["questions"])
                logits_list.append(outputs[q_id]["logits"])
                # ... collect targets
        
        logits = torch.cat(logits_list)
        targets = torch.cat(targets_list)
        
        # Optimize temperature
        temp = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.LBFGS([temp], lr=0.01, max_iter=100)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / temp
            loss = nn.CrossEntropyLoss()(scaled_logits, targets)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        temps[q_id] = temp.item()
    
    # Apply learned temperatures
    for q_id, temp in temps.items():
        model.temperatures[q_id] = temp
```

### Phase 1f: Expected results

With 10K–100K labeled examples and a frozen MiniLM encoder:

- **Yes/No accuracy:** 85–95% on in-domain tasks (depends on domain difficulty)
- **Choice accuracy (4–7 options):** 75–90% on in-domain tasks
- **Score accuracy (4–5 level Likert):** 70–85% (measured as within-1-level accuracy)
- **Calibration (ECE):** 0.05–0.15 after temperature scaling (lower is better; <0.05 is good)
- **Inference latency:** 50–200ms per query on CPU, 10–50ms on GPU

These are rough estimates. Actual results depend heavily on domain, data quality, and question complexity.

---

## Tier 2: Needle3 Distillation

### Goal

Compress the Tier 1 system's knowledge into a Needle3 model. The result is a 8–29 MB quantized model that runs on edge devices with grammar-constrained output.

### Phase 2a: Prepare distillation data

Use the trained Tier 1 model to label a large dataset:

```python
def generate_distillation_data(tier1_model, unlabeled_states, question_specs, output_path):
    """
    Use Tier 1 model to generate labeled data for distillation.
    """
    tier1_model.eval()
    data = []
    
    with torch.no_grad():
        for state in unlabeled_states:
            outputs = tier1_model([state], question_specs)
            example = {
                "state": state,
                "questions": question_specs,
                "answers": construct_answers(outputs),
                "tier1_confidence": {qid: out["confidence"].item() 
                                      for qid, out in outputs.items()},
            }
            data.append(example)
    
    save_json(data, output_path)
    return data
```

**Data volume:** 100K–10M examples for effective distillation. More is better, especially for rare question types and edge cases.

**Data diversity:** Cover the full range of decision difficulty — easy cases (high confidence, clear answer) and hard cases (low confidence, ambiguous). The model needs to learn when to be uncertain.

### Phase 2b: Needle3 fine-tuning setup

Needle3's Python SDK supports fine-tuning with LoRA:

```python
import needle

# Load base Needle3 model
base_model = needle.load("Cactus-Compute/needle3")

# Prepare data in Needle3's format
# Needle3 expects: { "text": state, "tools": [question_spec_as_tool], "expected": decision }
training_data = convert_to_needle3_format(distillation_data)

# LoRA fine-tuning at full 20 layers
adapter = needle.finetune(
    base_model,
    training_data,
    method="lora",
    epochs=3,
    lr=1e-4,
    batch_size=32,
)
```

**Adapting Needle3 for decision tasks:**

Needle3's default heads are for tool-calling and extraction. For general decisions:

1. **Define decision tools:** Frame each decision question as a "tool" with typed arguments:
   ```python
   @needle.tool
   def team_decision(state: str) -> Literal["billing", "technical", "sales", "account"]:
       """Which team should handle this?"""
       ...
   ```
   
2. **Use extraction for scores:** Frame scoring as extracting a structured record:
   ```python
   class FrustrationScore(BaseModel):
       level: Literal["calm", "mildly_annoyed", "frustrated", "angry_churn"]
       value: float  # 0–4
       confidence: float  # 0–1
   ```

3. **Yes/No as binary extraction:** Frame as extracting a boolean field.

**Alternative:** Modify Needle3's head architecture directly (requires working with the Cactus code). This gives more control but more complexity.

### Phase 2c: Distillation loss

```python
def distillation_loss(needle_output, tier1_output, temperature=2.0):
    """
    KL divergence between Needle3 and Tier 1 output distributions,
    plus confidence matching.
    """
    loss = 0.0
    
    for q_id in needle_output:
        n_probs = needle_output[q_id]["probabilities"]  # [batch, K]
        t_probs = tier1_output[q_id]["probabilities"]   # [batch, K]
        
        # KL divergence (softened with temperature)
        n_soft = torch.softmax(n_probs / temperature, dim=-1)
        t_soft = torch.softmax(t_probs / temperature, dim=-1)
        kl = (t_soft * (torch.log(t_soft + 1e-10) - torch.log(n_soft + 1e-10))).sum(dim=-1).mean()
        
        loss += kl * (temperature ** 2)  # scale by T^2 (Hinton et al. distillation)
        
        # Confidence matching
        n_conf = needle_output[q_id]["confidence"]
        t_conf = tier1_output[q_id]["confidence"]
        conf_loss = nn.MSELoss()(n_conf, t_conf)
        loss += 0.1 * conf_loss
    
    return loss
```

### Phase 2d: Calibration fine-tuning

After distillation, fine-tune the calibration:

```python
def calibration_finetuning(needle_model, cal_data):
    """
    Fine-tune Needle3's confidence head on held-out calibration data.
    Uses Brier score as the loss (proper scoring rule).
    """
    for example in cal_data:
        state = example["state"]
        questions = example["questions"]
        truth = example["answers"]  # ground truth decisions
        
        output = needle_model(state, questions)
        
        loss = 0.0
        for q_id, out in output.items():
            gt = truth[q_id]
            # One-hot encode ground truth
            if q_id type == "yes_no":
                true_dist = torch.tensor([0.0, 1.0]) if gt["yes_no"] == "yes" else torch.tensor([1.0, 0.0])
            elif q_id type == "choice":
                true_dist = ...  # one-hot over options
            else:  # score
                true_dist = ...  # one-hot over levels or soft distribution
            
            # Brier score
            brier = ((out["probabilities"] - true_dist) ** 2).sum(dim=-1)
            loss += brier
        
        # Update model (backprop through Needle3)
        loss.backward()
        optimizer.step()
```

### Phase 2e: Export and quantize

```python
# Merge LoRA adapter
merged = needle.merge(adapter)

# Slice to desired depth (2–20 layers)
sliced = needle.slice(merged, num_layers=8)  # 8-layer subnetwork

# Export to .cact format
needle.build(
    model=sliced,
    platform="linux-x86_64",  # or arm64, wasm, etc.
    output="system-one-8l.cact",
    quantization="cq2",  # 2.125 bits per weight
)
```

**Expected output sizes (CQ2 quantized):**
- 2 layers: ~8 MB
- 4 layers: ~12 MB
- 8 layers: ~18 MB
- 12 layers: ~23 MB
- 16 layers: ~26 MB
- 20 layers: ~29 MB

**Expected inference speed (Raspberry Pi 5):**
- 2–20K tokens/s prefill depending on layer count
- 400–4K tokens/s decode

For decision tasks, the "token" count is much lower than for text generation — you're encoding a state (maybe 100–500 tokens) and producing decisions (no generation). So actual latency is dominated by prefill, which should be well under 100ms for typical state sizes.

---

## Tier 3: RLCD-Style End-to-End Training

### Goal

Train a custom model architecture from scratch (or from a pretrained encoder) with a calibration-aware reinforcement learning objective.

### Phase 3a: Architecture

Use the architecture described in [`DESIGN.md`](DESIGN.md) — encoder + question interaction + parallel heads + calibration head.

### Phase 3b: Reward function

The core of RLCD is the reward function. Based on TypeSafe's description and the research literature:

```python
def rlcd_reward(model_output, ground_truth):
    """
    RLCD-style reward: accuracy + calibration.
    
    The key insight: a model should be rewarded for being
    (a) correct and (b) appropriately confident.
    A model that's 95% confident and right gets more reward
    than one that's 55% confident and right.
    A model that's 95% confident and wrong gets heavily penalized.
    """
    reward = 0.0
    
    for q_id, output in model_output.items():
        gt = ground_truth[q_id]
        
        # Accuracy component
        predicted = output["probabilities"].argmax(dim=-1)
        correct = (predicted == gt["true_index"]).float()
        
        # Calibration component (Brier score-based)
        true_dist = one_hot(gt["true_index"], output["probabilities"].shape[-1])
        brier = ((output["probabilities"] - true_dist) ** 2).sum(dim=-1)
        calibration = -brier  # negative Brier = good calibration
        
        # Combined reward
        # Weight calibration more when the model is confident
        conf_weight = 1.0 + output["confidence"]  # more confident → calibration matters more
        question_reward = correct + conf_weight * calibration
        
        reward += question_reward
    
    return reward / len(model_output)  # average across questions
```

**Why this works:** The Brier score is a proper scoring rule — it's minimized when the predicted probabilities equal the true probabilities. This means the model is incentivized to output its true belief, not to game the reward by being overconfident or underconfident.

### Phase 3c: Training with RL

```python
def train_rlcd(model, train_data, epochs=100, rl_steps_per_epoch=1000):
    """
    RLCD-style training loop.
    
    Uses REINFORCE-style policy gradient on the decision outputs.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        model.train()
        rewards = []
        
        for step in range(rl_steps_per_epoch):
            # Sample a batch
            batch = sample_batch(train_data)
            
            # Forward pass
            outputs = model(batch["states"], batch["questions"])
            
            # Compute reward
            reward = rlcd_reward(outputs, batch["answers"])
            
            # Policy gradient loss
            # log_prob of the chosen action (decision) * reward
            log_probs = compute_log_probs(outputs)  # log P(choice | state, question)
            loss = -(log_probs * reward.detach()).mean()  # REINFORCE
            
            # Add supervised loss as baseline (helps stability)
            supervised_loss = cross_entropy_loss(outputs, batch["answers"])
            loss = 0.5 * loss + 0.5 * supervised_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            rewards.append(reward.item())
        
        # Evaluate calibration
        ece = evaluate_calibration(model, val_data)
        print(f"Epoch {epoch}: avg_reward={np.mean(rewards):.4f}, ECE={ece:.4f}")
```

### Phase 3d: Group Relative Policy Optimization (GRPO) variant

GRPO (from DeepSeek-R1) is a more stable alternative to vanilla REINFORCE:

```python
def train_grpo(model, train_data, group_size=4):
    """
    GRPO-style training: sample multiple outputs per input,
    compute relative advantages within the group.
    """
    for batch in train_data:
        states = batch["states"]
        questions = batch["questions"]
        truths = batch["answers"]
        
        # Sample G outputs for each input
        group_outputs = []
        for _ in range(group_size):
            output = model(states, questions)
            group_outputs.append(output)
        
        # Compute rewards for each member of the group
        rewards = [rlcd_reward(out, truths) for out in group_outputs]
        
        # Compute advantages (relative to group mean)
        reward_tensor = torch.stack(rewards)
        mean_reward = reward_tensor.mean(dim=0)
        std_reward = reward_tensor.std(dim=0) + 1e-8
        advantages = (reward_tensor - mean_reward) / std_reward
        
        # Policy gradient with relative advantages
        loss = 0.0
        for g in range(group_size):
            log_probs = compute_log_probs(group_outputs[g])
            loss -= (log_probs * advantages[g].detach()).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Phase 3e: Practical considerations

**Exploration:** RL needs exploration. For decision models, exploration means trying different probability distributions and seeing which get higher rewards. This is harder than token-level exploration because the action space is continuous (probability distributions).

**Stability:** RL training is notoriously unstable. Start with a strong supervised baseline, then add RL fine-tuning with a small learning rate. Use the supervised loss as a regularizer.

**Reward shaping:** The raw Brier + accuracy reward may not be enough. Consider:
- Rewarding consistency (similar inputs → similar outputs)
- Rewarding calibration on the calibration set specifically
- Penalizing variance across multiple samples of the same input

**Compute requirements:**
- Tier 3 training is significantly more compute-intensive than Tier 1 or 2
- Expect to need 1–4 GPUs for days to weeks depending on model size and data volume
- The RL loop adds overhead beyond standard supervised training

---

## Training data summary

| Tier | Data type | Volume | Source |
|------|-----------|--------|--------|
| **Tier 1** | Labeled (state, questions, decisions) | 10K–100K | Open datasets + synthetic + hand-labeled |
| **Tier 2** | Distillation data from Tier 1 | 100K–10M | Generated by Tier 1 model |
| **Tier 3** | Verifiable (state, questions, truth) | 100K–10M+ | Open datasets + programmatic verification + synthetic |

See [`DATASETS.md`](DATASETS.md) for detailed dataset inventory and construction recipes.
