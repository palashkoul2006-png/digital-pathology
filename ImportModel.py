# ImportModel.py - Skin Cancer Detection using ViT
# Compatible with transformers v5.x

from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from PIL import Image
import requests

print("✅ All libraries imported successfully!")

# -----------------------------------------------
# STEP 1: Load the Pretrained Model & Processor
# -----------------------------------------------
model_name = "google/vit-base-patch16-224"

print(f"⏳ Loading model: {model_name} ...")

processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

model.eval()  # Set model to evaluation mode

print("✅ Model loaded successfully!")
print(f"📌 Model: {model_name}")
print(f"📌 Number of labels: {model.config.num_labels}")

# -----------------------------------------------
# STEP 2: Test with a Sample Image (from URL)
# -----------------------------------------------
print("\n⏳ Testing with a sample image...")

# Download a test image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

try:
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    print("✅ Sample image loaded!")
except:
    # If no internet, create a blank test image
    image = Image.new("RGB", (224, 224), color=(128, 64, 32))
    print("✅ Created blank test image (no internet needed)!")

# -----------------------------------------------
# STEP 3: Preprocess & Run Inference
# -----------------------------------------------
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]

print(f"\n🎯 Predicted Class Index : {predicted_class_idx}")
print(f"🎯 Predicted Label       : {predicted_label}")

# -----------------------------------------------
# STEP 4: Show Model is Ready for Fine-tuning
# -----------------------------------------------
print("\n" + "="*50)
print("✅ MODEL IS READY!")
print("="*50)
print("Next steps:")
print("  1. Load your skin cancer dataset (HAM10000)")
print("  2. Fine-tune this model on 7 skin cancer classes")
print("  3. Train & evaluate your model")
print("="*50)