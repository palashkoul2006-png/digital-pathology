# train.py - Fine-tune ViT on Skin Cancer (HAM10000)

from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
import torch

print("⏳ Loading HAM10000 skin cancer dataset...")

# Load dataset directly from Hugging Face (no download needed!)
dataset = load_dataset("marmal88/skin_cancer")

print("✅ Dataset loaded!")
print(dataset)

# Skin cancer class labels
labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}

print("\n📌 Classes:", labels)

# Load model for 7 skin cancer classes
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=7,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True  # replaces old 1000-class head
)

print("✅ Model ready for skin cancer fine-tuning!")

# Save the model locally in your project
model.save_pretrained("./saved_model")
processor.save_pretrained("./saved_model")

print("✅ Model saved to ./saved_model folder!")