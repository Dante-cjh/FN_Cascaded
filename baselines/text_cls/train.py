"""
baselines/text_cls/train.py
Fine-tune a DeBERTa / RoBERTa text classifier for binary Fake/True detection.

Input:  data/processed/{basepack,augmented}_{train,val}.jsonl
Output: outputs/{exp_dir}/model/best_model/   (HuggingFace checkpoint)
        outputs/{exp_dir}/model/val_metrics.json

input_mode controls which text field is fed to the model:
  "base"              → basepack_text       (Exp-1: Small-Only)
  "base_plus_llm_aug" → augmented_text      (Exp-2: LLM-Pre + Small)

Usage (Exp-1):
  python baselines/text_cls/train.py \
      --train      data/processed/basepack_train.jsonl \
      --val        data/processed/basepack_val.jsonl \
      --input_mode base \
      --output_dir outputs/exp1_small_only/model

Usage (Exp-2):
  python baselines/text_cls/train.py \
      --train      data/processed/augmented_train.jsonl \
      --val        data/processed/augmented_val.jsonl \
      --input_mode base_plus_llm_aug \
      --output_dir outputs/exp2_llm_pre/model
"""

import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report


# --------------------------------------------------------------------------- #
#  Unified input builder
# --------------------------------------------------------------------------- #

def build_model_input(sample: dict, mode: str) -> str:
    """
    Build the text string fed to the model depending on experiment mode.

    mode="base"              → basepack_text  (Exp-1)
    mode="base_plus_llm_aug" → augmented_text (Exp-2); falls back to basepack_text
                               if LLM augmentation failed for this sample.
    """
    if mode == "base":
        return sample.get("basepack_text") or sample.get("source_text", "")
    elif mode == "base_plus_llm_aug":
        return (sample.get("augmented_text")
                or sample.get("basepack_text")
                or sample.get("source_text", ""))
    else:
        # Legacy / direct field name fallback
        return sample.get(mode) or sample.get("source_text", "")


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #

class RumorDataset(Dataset):
    def __init__(self, events: list[dict], tokenizer, max_length: int = 512,
                 input_mode: str = "base"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []
        for e in events:
            text = build_model_input(e, input_mode)
            label = e["label"]
            self.items.append((text, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text, label = self.items[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
#  Training loop
# --------------------------------------------------------------------------- #

def load_events(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(model, loader, device, label_names=None):
    if label_names is None:
        label_names = ["True", "Fake"]
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu()
            preds.extend(logits.argmax(-1).tolist())
            golds.extend(labels.tolist())
    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, average="macro", zero_division=0)
    report = classification_report(golds, preds, target_names=label_names, zero_division=0)
    return acc, f1, report


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    label_names = args.label_names if args.label_names else ["True", "Fake"]
    input_mode = args.input_mode

    print(f"Loading tokenizer: {args.model_name}")
    # use_fast=False: 强制使用 SentencePiece 慢速分词器，避免新版 transformers
    # 错误地用 tiktoken 解析 deberta-v3 的 spm.model 文件
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)

    train_events = load_events(Path(args.train))
    val_events = load_events(Path(args.val))
    print(f"Train: {len(train_events)}, Val: {len(val_events)}")
    print(f"input_mode: {input_mode}  |  labels: {label_names}")

    train_dataset = RumorDataset(train_events, tokenizer, args.max_length, input_mode)
    val_dataset   = RumorDataset(val_events,   tokenizer, args.max_length, input_mode)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_f1 = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            input_ids     = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels        = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 50 == 0:
                print(f"  Epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={total_loss/step:.4f}")

        acc, f1, report = evaluate(model, val_loader, device, label_names)
        print(f"Epoch {epoch}: val_acc={acc:.4f}  val_macro_f1={f1:.4f}")
        print(report)

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"  Saved best model (f1={best_f1:.4f}) -> {best_model_dir}")

    metrics = {
        "best_val_macro_f1": best_f1,
        "input_mode": input_mode,
        "label_names": label_names,
    }
    with open(output_dir / "val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nTraining complete. Best val macro-F1: {best_f1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",       default="data/processed/basepack_train.jsonl")
    parser.add_argument("--val",         default="data/processed/basepack_val.jsonl")
    parser.add_argument("--model_name",  default="microsoft/deberta-v3-base")
    parser.add_argument("--max_length",  type=int,   default=512)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--num_epochs",  type=int,   default=5)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--output_dir",  default="outputs/exp1_small_only/model")
    parser.add_argument(
        "--input_mode",
        default="base",
        choices=["base", "base_plus_llm_aug"],
        help=(
            "base              → basepack_text (Exp-1)\n"
            "base_plus_llm_aug → augmented_text (Exp-2)"
        ),
    )
    parser.add_argument(
        "--label_names", nargs=2, default=["True", "Fake"],
        help="Display names for label 0 and label 1",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
