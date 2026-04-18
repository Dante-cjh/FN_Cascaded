"""
04_predict_small_model.py
Run inference with the fine-tuned small model on a dataset split.

Supports both experiment modes via --input_mode:
  "base"              → basepack_text       (Exp-1)
  "base_plus_llm_aug" → augmented_text      (Exp-2)

Output: outputs/{exp_dir}/test_predictions.jsonl
Each line:
{
  "event_id":   str,
  "gold":       int,
  "pred":       int,
  "prob":       [float, float],   # [prob_label0, prob_label1]
  "confidence": float,            # max(prob)
  "margin":     float,            # |prob[1] - prob[0]|
  "pred_label": str,              # "True" or "Fake"
  "gold_label": str
}
"""

import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Reuse the same input builder as training (keeps modes consistent)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.text_cls.train import build_model_input


class RumorInferenceDataset(Dataset):
    def __init__(self, events: list[dict], tokenizer, max_length: int,
                 input_mode: str = "base"):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.events     = events
        self.input_mode = input_mode

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        e    = self.events[idx]
        text = build_model_input(e, self.input_mode)
        enc  = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "event_id":       e["event_id"],
            "gold":           e["label"],
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def load_events(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def predict(args):
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_names = args.label_names if args.label_names else ["True", "Fake"]
    input_mode  = args.input_mode
    print(f"Device: {device}  |  input_mode: {input_mode}")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: Model not found at {model_dir}. Run the training script first.")
        return

    print(f"Loading model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    events = load_events(Path(args.input))
    print(f"Running inference on {len(events)} events...")

    dataset = RumorInferenceDataset(events, tokenizer, args.max_length, input_mode)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            probs          = F.softmax(outputs.logits, dim=-1).cpu()

            for i in range(len(batch["event_id"])):
                prob       = probs[i].tolist()
                pred       = int(probs[i].argmax().item())
                gold       = int(batch["gold"][i].item())
                confidence = max(prob)
                margin     = abs(prob[1] - prob[0])
                results.append({
                    "event_id":   batch["event_id"][i],
                    "gold":       gold,
                    "pred":       pred,
                    "prob":       prob,
                    "confidence": confidence,
                    "margin":     margin,
                    "pred_label": label_names[pred] if pred < len(label_names) else str(pred),
                    "gold_label": label_names[gold] if gold < len(label_names) else str(gold),
                })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # 推理完成后释放显存
    del model
    torch.cuda.empty_cache()

    # Quick metrics
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    golds = [r["gold"] for r in results]
    preds = [r["pred"] for r in results]
    acc = accuracy_score(golds, preds)
    f1  = f1_score(golds, preds, average="macro", zero_division=0)
    print(f"\nResults on {args.input}:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro-F1 : {f1:.4f}")
    print(classification_report(golds, preds, target_names=label_names, zero_division=0))
    print(f"Predictions saved -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="data/processed/basepack_test.jsonl")
    parser.add_argument("--model_dir",  default="outputs/exp1_small_only/model/best_model")
    parser.add_argument("--output",     default="outputs/exp1_small_only/test_predictions.jsonl")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--input_mode",
        default="base",
        choices=["base", "base_plus_llm_aug"],
        help="base=basepack_text (Exp-1)  |  base_plus_llm_aug=augmented_text (Exp-2)",
    )
    parser.add_argument(
        "--label_names", nargs=2, default=["True", "Fake"],
        help="Display names for label 0 and label 1",
    )
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
