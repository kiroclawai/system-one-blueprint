"""
Tier 1 Reference Implementation: System One Model
Encoder + Decision Heads

A working reference implementation of a Jev-class decision model using
a pretrained encoder (MiniLM) + parallel decision heads.

Usage:
    python src/tier1_model.py train --data data/train.jsonl --epochs 10
    python src/tier1_model.py evaluate --model best_model.pt --data data/test.jsonl
    python src/tier1_model.py serve --model best_model.pt
"""

import json
import math
import time
import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss


# ──────────────────────────────────────────────
# Question specification
# ──────────────────────────────────────────────

class QuestionSpec:
    """A typed decision question."""
    def __init__(self, qid: str, qtype: str, options: list[str] | None = None,
                 scale: list[str] | None = None, description: str = ""):
        self.id = qid
        self.type = qtype  # "yes_no" | "choice" | "score"
        self.options = options or []
        self.scale = scale or []
        self.description = description
    
    def num_outputs(self) -> int:
        if self.type == "yes_no":
            return 2
        elif self.type == "choice":
            return len(self.options)
        elif self.type == "score":
            return len(self.scale)
        return 2
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "options": self.options,
            "scale": self.scale,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "QuestionSpec":
        return cls(d["id"], d["type"], d.get("options"), d.get("scale"), d.get("description", ""))


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class SystemOneDataset(Dataset):
    """Dataset of (state, questions, answers) triples."""
    
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, 
                 max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        state = ex["state"]
        questions = [QuestionSpec.from_dict(q) for q in ex["questions"]]
        answers = ex["answers"]
        
        # Tokenize state
        tokens = self.tokenizer(
            state,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Build question tensors
        question_tensors = []
        answer_indices = []
        answer_confidences = []
        
        for q in questions:
            # Encode question description
            q_tokens = self.tokenizer(
                q.description,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            question_tensors.append(q_tokens["input_ids"].squeeze(0))
            
            # Get answer index
            ans = answers[q.id]
            if q.type == "yes_no":
                idx = 1 if ans["yes_no"] == "yes" else 0
            elif q.type == "choice":
                idx = q.options.index(ans["choice"])
            elif q.type == "score":
                idx = q.scale.index(ans["answer"])
            else:
                idx = 0
            
            answer_indices.append(idx)
            answer_confidences.append(ans.get("confidence", 1.0))
        
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "question_input_ids": torch.stack(question_tensors),
            "answer_indices": torch.tensor(answer_indices, dtype=torch.long),
            "answer_confidences": torch.tensor(answer_confidences, dtype=torch.float),
            "questions": questions,
        }


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class DecisionHead(nn.Module):
    """A single decision head for one question."""
    
    def __init__(self, input_dim: int, num_outputs: int):
        super().__init__()
        self.num_outputs = num_outputs
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_outputs),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SystemOneTier1(nn.Module):
    """
    Tier 1 System One model.
    
    Architecture:
    - Pretrained encoder (frozen or tunable)
    - Question embedding via encoder
    - CLS pooling + question concat → per-question representation
    - Parallel decision heads (one per question)
    - Temperature scaling per head (for calibration)
    """
    
    def __init__(self, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 questions: list[QuestionSpec] | None = None,
                 freeze_encoder: bool = True,
                 calibration: bool = True):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        hidden_dim = self.encoder.config.hidden_size
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Question embedding: encode description with encoder, pool to fixed dim
        self.question_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Pooling: concat CLS + question embedding
        self.pooler = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Decision heads
        self.questions = questions or []
        self.heads = nn.ModuleDict()
        self.temperatures = nn.ParameterDict()
        
        if questions:
            for q in questions:
                num_out = q.num_outputs()
                self.heads[q.id] = DecisionHead(hidden_dim, num_out)
                if calibration:
                    self.temperatures[q.id] = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                question_input_ids: torch.Tensor) -> dict[str, dict]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len] state text tokens
            attention_mask: [batch, seq_len]
            question_input_ids: [num_questions, q_seq_len] question tokens
        
        Returns:
            dict mapping question_id → {
                "logits": [batch, num_outputs],
                "probabilities": [batch, num_outputs],
                "confidence": [batch],
            }
        """
        batch_size = input_ids.shape[0]
        num_questions = question_input_ids.shape[0]
        device = input_ids.device
        
        # Encode state
        state_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = state_out.last_hidden_state[:, 0, :]  # [batch, hidden_dim]
        
        # Encode each question
        question_embeddings = []
        for q_idx in range(num_questions):
            q_ids = question_input_ids[q_idx:q_idx+1].expand(batch_size, -1)
            # Simple mean pooling over question tokens
            with torch.no_grad():  # question encoder is frozen
                q_out = self.encoder(input_ids=q_ids)
                q_repr = q_out.last_hidden_state.mean(dim=1)  # [batch, hidden_dim]
            q_repr = self.question_proj(q_repr)
            question_embeddings.append(q_repr)
        
        question_embeddings = torch.stack(question_embeddings, dim=1)  # [batch, num_q, hidden]
        
        # Pool and predict for each question
        outputs = {}
        for i, q in enumerate(self.questions):
            q_emb = question_embeddings[:, i, :]  # [batch, hidden]
            combined = torch.cat([cls_repr, q_emb], dim=-1)  # [batch, hidden*2]
            pooled = self.pooler(combined)  # [batch, hidden]
            
            logits = self.heads[q.id](pooled)  # [batch, num_outputs]
            
            # Apply temperature
            if q.id in self.temperatures:
                temp = self.temperatures[q.id]
                logits = logits / temp
            
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values
            
            outputs[q.id] = {
                "logits": logits,
                "probabilities": probs,
                "confidence": confidence,
            }
        
        return outputs


# ──────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────

def compute_loss(outputs: dict, answer_indices: torch.Tensor,
                 answer_confidences: torch.Tensor, questions: list[QuestionSpec],
                 calibration_weight: float = 0.1) -> tuple[torch.Tensor, dict]:
    """
    Compute combined loss across all questions.
    
    Loss = CrossEntropy + calibration_penalty
    
    The calibration penalty encourages the model to be confident when right
    and uncertain when wrong.
    """
    total_loss = torch.tensor(0.0, device=answer_indices.device)
    losses = {}
    
    for i, q in enumerate(questions):
        q_out = outputs[q.id]
        logits = q_out["logits"]
        probs = q_out["probabilities"]
        confidence = q_out["confidence"]
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, answer_indices[:, i])
        
        # Calibration penalty: |confidence - correctness|
        correct = (probs.argmax(dim=-1) == answer_indices[:, i]).float()
        cal_penalty = torch.abs(confidence - correct).mean()
        
        loss = ce_loss + calibration_weight * cal_penalty
        total_loss += loss
        losses[q.id] = {"ce": ce_loss.item(), "cal": cal_penalty.item(), "total": loss.item()}
    
    return total_loss, losses


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train(model: SystemOneTier1, train_path: str, val_path: str | None,
          epochs: int = 10, batch_size: int = 32, lr: float = 1e-3,
          weight_decay: float = 0.01, lr_scheduler: bool = True,
          checkpoint_dir: str = "checkpoints"):
    """
    Train the Tier 1 model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_ds = SystemOneDataset(train_path, model.tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=2, pin_memory=True)
    
    val_loader = None
    if val_path:
        val_ds = SystemOneDataset(val_path, model.tokenizer)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)
    
    # Optimizer: only train non-frozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    
    scheduler = None
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            question_ids = batch["question_input_ids"].to(device)
            answer_indices = batch["answer_indices"].to(device)
            answer_confidences = batch["answer_confidences"].to(device)
            questions = batch["questions"]
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask, question_ids)
            loss, _ = compute_loss(outputs, answer_indices, answer_confidences, questions)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        train_loss /= max(train_steps, 1)
        
        # Validation
        val_loss = None
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    question_ids = batch["question_input_ids"].to(device)
                    answer_indices = batch["answer_indices"].to(device)
                    answer_confidences = batch["answer_confidences"].to(device)
                    questions = batch["questions"]
                    
                    outputs = model(input_ids, attention_mask, question_ids)
                    loss, _ = compute_loss(outputs, answer_indices, answer_confidences, questions)
                    
                    val_loss += loss.item()
                    val_steps += 1
            
            val_loss /= max(val_steps, 1)
        
        # Learning rate scheduler
        if scheduler:
            scheduler.step()
        
        # Checkpoint
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "questions": [q.to_dict() for q in model.questions],
            }, Path(checkpoint_dir) / "best_model.pt")
        
        val_str = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}{val_str}")
    
    # Load best model
    best_path = Path(checkpoint_dir) / "best_model.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1} "
              f"(val_loss={checkpoint['val_loss']:.4f})")
    
    return model


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    
    Bin predictions by confidence, measure accuracy gap in each bin.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        bin_size = in_bin.sum()
        
        if bin_size > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += (bin_size / len(confidences)) * abs(avg_accuracy - avg_confidence)
    
    return ece


def evaluate(model: SystemOneTier1, data_path: str, batch_size: int = 32) -> dict:
    """
    Evaluate the model on a dataset.
    
    Returns dict with accuracy, ECE, Brier score per question type.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    ds = SystemOneDataset(data_path, model.tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Collect per-question results
    results = {}  # q_id → {accuracies, confidences, probs, labels}
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            question_ids = batch["question_input_ids"].to(device)
            answer_indices = batch["answer_indices"].to(device)
            questions = batch["questions"]
            
            outputs = model(input_ids, attention_mask, question_ids)
            
            for i, q in enumerate(questions):
                q_id = q.id
                if q_id not in results:
                    results[q_id] = {
                        "probs": [], "labels": [], "confidences": [],
                        "qtype": q.type, "options": q.options,
                    }
                
                probs = outputs[q_id]["probabilities"].cpu().numpy()
                confs = outputs[q_id]["confidence"].cpu().numpy()
                labels = answer_indices[:, i].cpu().numpy()
                
                results[q_id]["probs"].append(probs)
                results[q_id]["labels"].append(labels)
                results[q_id]["confidences"].append(confs)
    
    # Aggregate metrics
    metrics = {}
    for q_id, data in results.items():
        probs = np.vstack(data["probs"])
        labels = np.concatenate(data["labels"])
        confidences = np.concatenate(data["confidences"])
        
        predictions = probs.argmax(axis=1)
        accuracy = accuracy_score(labels, predictions)
        ece = compute_ece(probs, labels)
        
        # Brier score
        n_classes = probs.shape[1]
        labels_one_hot = np.eye(n_classes)[labels]
        brier = brier_score_loss(labels_one_hot.ravel(), probs.ravel())
        
        # Overconfidence: mean confidence - accuracy
        overconfidence = confidences.mean() - accuracy
        
        metrics[q_id] = {
            "type": data["qtype"],
            "num_classes": n_classes,
            "accuracy": accuracy,
            "ece": ece,
            "brier": brier,
            "overconfidence": overconfidence,
            "mean_confidence": confidences.mean(),
            "predictions": predictions.tolist(),
            "labels": labels.tolist(),
        }
    
    # Aggregate across all questions
    all_probs = np.vstack([v["probs"][0] for v in results.values() 
                           for _ in range(len(v["probs"]))])
    # Actually aggregate properly
    all_probs_list = []
    all_labels_list = []
    for q_id, data in results.items():
        all_probs_list.append(np.vstack(data["probs"]))
        all_labels_list.append(np.concatenate(data["labels"]))
    
    all_probs = np.concatenate(all_probs_list)
    all_labels = np.concatenate(all_labels_list)
    
    overall = {
        "accuracy": accuracy_score(all_labels, all_probs.argmax(axis=1)),
        "ece": compute_ece(all_probs, all_labels),
        "brier": brier_score_loss(
            np.eye(all_probs.shape[1])[all_labels].ravel(),
            all_probs.ravel()
        ),
    }
    
    return {"per_question": metrics, "overall": overall}


# ──────────────────────────────────────────────
# Output construction
# ──────────────────────────────────────────────

def construct_output(questions: list[QuestionSpec], 
                     model_outputs: dict) -> dict:
    """
    Construct typed decision output from model predictions.
    
    Guarantees valid output matching the question schema.
    """
    result = {}
    
    for q in questions:
        out = model_outputs[q.id]
        probs = out["probabilities"]
        
        if q.type == "yes_no":
            yes_prob = probs[1].item()  # index 1 = "yes"
            result[q.id] = {
                "type": "yes_no",
                "answer": "yes" if yes_prob >= 0.5 else "no",
                "probability": round(yes_prob, 4),
                "confidence": round(out["confidence"].item(), 4),
                "probabilities": {
                    "yes": round(probs[1].item(), 4),
                    "no": round(probs[0].item(), 4),
                },
            }
        
        elif q.type == "choice":
            probs_np = probs.detach().cpu().numpy() if isinstance(probs, torch.Tensor) else probs
            best_idx = int(probs_np.argmax())
            result[q.id] = {
                "type": "choice",
                "answer": q.options[best_idx],
                "confidence": round(float(probs_np[best_idx]), 4),
                "probabilities": {
                    opt: round(float(probs_np[i]), 4) 
                    for i, opt in enumerate(q.options)
                },
            }
        
        elif q.type == "score":
            probs_np = probs.detach().cpu().numpy() if isinstance(probs, torch.Tensor) else probs
            best_idx = int(probs_np.argmax())
            value = float(best_idx)  # ordinal value
            result[q.id] = {
                "type": "score",
                "answer": q.scale[best_idx],
                "value": round(value, 2),
                "confidence": round(float(probs_np[best_idx]), 4),
                "probabilities": {
                    level: round(float(probs_np[i]), 4)
                    for i, level in enumerate(q.scale)
                },
            }
    
    return result


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="System One Tier 1 Model")
    subparsers = parser.add_subparsers(dest="command")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", required=True, help="Path to training JSONL")
    train_parser.add_argument("--val-data", help="Path to validation JSONL")
    train_parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--checkpoint-dir", default="checkpoints")
    train_parser.add_argument("--questions", help="Path to questions JSON")
    train_parser.add_argument("--freeze-encoder", action="store_true", default=True)
    
    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--model", required=True, help="Path to model checkpoint")
    eval_parser.add_argument("--data", required=True, help="Path to evaluation JSONL")
    eval_parser.add_argument("--batch-size", type=int, default=32)
    
    # Serve
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--model", required=True, help="Path to model checkpoint")
    serve_parser.add_argument("--port", type=int, default=8080)
    
    args = parser.parse_args()
    
    if args.command == "train":
        # Load questions
        questions = []
        if args.questions:
            with open(args.questions) as f:
                for q in json.load(f):
                    questions.append(QuestionSpec.from_dict(q))
        
        model = SystemOneTier1(
            encoder_name=args.encoder,
            questions=questions,
            freeze_encoder=args.freeze_encoder,
        )
        
        train(model, args.data, args.val_data, args.epochs, args.batch_size,
              args.lr, checkpoint_dir=args.checkpoint_dir)
    
    elif args.command == "evaluate":
        checkpoint = torch.load(args.model, map_location="cpu")
        questions = [QuestionSpec.from_dict(q) for q in checkpoint["questions"]]
        
        model = SystemOneTier1(encoder_name="sentence-transformers/all-MiniLM-L6-v2",
                               questions=questions, freeze_encoder=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        metrics = evaluate(model, args.data, args.batch_size)
        
        print("\n=== Evaluation Results ===")
        print(f"\nOverall:")
        print(f"  Accuracy: {metrics['overall']['accuracy']:.4f}")
        print(f"  ECE:      {metrics['overall']['ece']:.4f}")
        print(f"  Brier:    {metrics['overall']['brier']:.4f}")
        
        print(f"\nPer-question:")
        for q_id, m in metrics["per_question"].items():
            print(f"  {q_id} ({m['type']}, {m['num_classes']} classes):")
            print(f"    Accuracy:       {m['accuracy']:.4f}")
            print(f"    ECE:            {m['ece']:.4f}")
            print(f"    Brier:          {m['brier']:.4f}")
            print(f"    Overconfidence: {m['overconfidence']:.4f}")
            print(f"    Mean confidence: {m['mean_confidence']:.4f}")
    
    elif args.command == "serve":
        checkpoint = torch.load(args.model, map_location="cpu")
        questions = [QuestionSpec.from_dict(q) for q in checkpoint["questions"]]
        
        model = SystemOneTier1(encoder_name="sentence-transformers/all-MiniLM-L6-v2",
                               questions=questions, freeze_encoder=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # Simple HTTP server
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json as json_mod
        
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers["Content-Length"])
                body = self.rfile.read(content_length)
                request = json_mod.loads(body)
                
                start = time.time()
                
                with torch.no_grad():
                    input_ids = torch.tensor([model.tokenizer(request["state"],
                        truncation=True, max_length=512, return_tensors="pt")["input_ids"]]).squeeze(0)
                    attention_mask = torch.tensor([model.tokenizer(request["state"],
                        truncation=True, max_length=512, return_tensors="pt")["attention_mask"]]).squeeze(0)
                    
                    q_ids = torch.stack([
                        model.tokenizer(q["description"], truncation=True, max_length=64, 
                                       return_tensors="pt")["input_ids"].squeeze(0)
                        for q in request["questions"]
                    ])
                    
                    outputs = model(input_ids, attention_mask, q_ids)
                    decisions = construct_output(questions, outputs)
                
                latency = (time.time() - start) * 1000
                
                response = json_mod.dumps({
                    "decisions": decisions,
                    "latency_ms": round(latency, 2),
                })
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response.encode())
            
            def log_message(self, format, *args):
                pass  # Suppress default logging
        
        server = HTTPServer(("0.0.0.0", args.port), Handler)
        print(f"Serving on port {args.port}...")
        server.serve_forever()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
