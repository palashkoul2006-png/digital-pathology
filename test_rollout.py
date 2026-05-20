"""
test_rollout.py — Quick standalone test of the Attention Rollout heatmap.
Saves output to static/uploads/rollout_test.jpg so you can inspect it.
"""
import sys, time
import cv2, numpy as np

# Load model (same way app.py does)
from models import load_models
from heatmaps import generate_heatmap
from utils import preprocess_image

IMG = r"test_images\test_0_actinic_keratoses.jpg"
OUT = r"static\uploads\rollout_test.jpg"

print("Loading model on GPU...")
t0 = time.time()
models = load_models()
vit = models['vit']
print(f"Model ready in {time.time()-t0:.1f}s")

print("Preprocessing image...")
arr = preprocess_image(IMG)

print("Generating Attention Rollout heatmap...")
t1 = time.time()
result = generate_heatmap(vit, arr, IMG)
elapsed = time.time() - t1
print(f"Heatmap generated in {elapsed:.2f}s")

if result.startswith("data:image/jpeg;base64,"):
    import base64
    raw = base64.b64decode(result.split(",", 1)[1])
    with open(OUT, "wb") as f:
        f.write(raw)
    print(f"Saved to {OUT}")
    print("SUCCESS — open the file to visually inspect the rollout heatmap.")
else:
    print("ERROR: unexpected result format")
    sys.exit(1)
