# finetune.py - FIXED VERSION with GPU + correct labels

from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ── Check GPU ────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Using device: {device}")
if device == "cuda":
    print(f"⚡ GPU: {torch.cuda.get_device_name(0)}")

# ── Load Dataset FIRST to get real label names ───────
print("⏳ Loading dataset...")
dataset = load_dataset("marmal88/skin_cancer")
print("✅ Dataset loaded!")

# ── Get ACTUAL label names from dataset ──────────────
actual_labels = sorted(dataset["train"].unique("dx"))
print(f"📌 Actual Classes Found: {actual_labels}")

label2id = {l: i for i, l in enumerate(actual_labels)}
id2label  = {i: l for i, l in enumerate(actual_labels)}
num_labels = len(actual_labels)
print(f"📌 Total Classes: {num_labels}")

# ── Load Processor & Model ───────────────────────────
print("⏳ Loading model...")
model_name = "google/vit-base-patch16-224"
processor  = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels           = num_labels,
    id2label             = id2label,
    label2id             = label2id,
    ignore_mismatched_sizes = True
)
model.to(device)
print("✅ Model loaded and moved to GPU!")

# ── Preprocess Images ────────────────────────────────
def preprocess(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    inputs = processor(images=images, return_tensors="pt")
    inputs["labels"] = [label2id[dx] for dx in batch["dx"]]
    return inputs

print("⏳ Preprocessing dataset (this takes ~5 mins)...")
dataset = dataset.map(
    preprocess,
    batched=True,
    batch_size=32,
    remove_columns=dataset["train"].column_names
)
dataset.set_format("torch")
print("✅ Preprocessing done!")

# ── Metrics ──────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# ── Training Arguments ───────────────────────────────
args = TrainingArguments(
    output_dir                  = "./skin_cancer_vit",
    num_train_epochs            = 10,
    per_device_train_batch_size = 32,   # bigger batch for GPU
    per_device_eval_batch_size  = 64,
    learning_rate               = 2e-5,
    warmup_ratio                = 0.1,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "accuracy",
    logging_steps               = 30,
    fp16                        = True,  # ⚡ GPU half precision ON
    report_to                   = "none",
)

# ── Trainer ──────────────────────────────────────────
trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = dataset["train"],
    eval_dataset    = dataset["validation"],
    compute_metrics = compute_metrics,
)

# ── Train! ───────────────────────────────────────────
print("\n🚀 Training started! (GPU mode)")
trainer.train()

# ── Save Final Model ─────────────────────────────────
model.save_pretrained("./saved_model")
processor.save_pretrained("./saved_model")
print("\n✅ Training complete! Model saved to ./saved_model")

# ── Evaluate on Test Set ─────────────────────────────
print("\n⏳ Evaluating on test set...")
results = trainer.evaluate(dataset["test"])
print(f"\n🎯 Test Accuracy : {results['eval_accuracy']*100:.2f}%")
print(f"🎯 Test F1 Score : {results['eval_f1']*100:.2f}%")