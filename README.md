# PathAI — Digital Pathology Image Analysis

An AI-powered medical image screening web application built with  
**Flask + Vision Transformer (ViT)** fine-tuned on the HAM10000 skin lesion dataset.  
Achieves **98.52% accuracy** across 7 dermatological classes.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Flask 3.x (Python 3.11) |
| **AI Model** | `google/vit-base-patch16-224` fine-tuned on HAM10000 |
| **Inference** | PyTorch 2.7 + CUDA (GPU) or CPU fallback |
| **Heatmap** | ViT Attention Rollout (real transformer attention) |
| **Frontend** | Vanilla HTML/CSS/JS — glassmorphism dark UI |

---

## Project Structure

```
digital_pathology/
├── app.py               # Flask routes & entry point
├── models.py            # ViT model loading & 7-class inference
├── heatmaps.py          # Attention Rollout → Grad-CAM → simulated fallback
├── utils.py             # Image pre-processing, H&E detection, recommendations
├── finetune.py          # Fine-tuning script (HAM10000, used offline)
├── requirements.txt
├── saved_model/         # Fine-tuned ViT weights (model.safetensors + config)
├── test_images/         # Sample HAM10000 test images
├── templates/
│   └── index.html       # Full UI (glassmorphism, animated bg, camera, results)
└── static/
    ├── style.css        # Animated CSS background, glassmorphism panels
    ├── script.js        # Tab switching, drag-drop, camera, results rendering
    └── uploads/         # Auto-created — stores temp uploaded images
```

---

## Quick Start

### 1. Clone / download the project

```bash
cd digital_pathology
```

### 2. Create & activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install GPU-enabled PyTorch (recommended)

> Skip this if running on CPU — the app works, just slower (~4 min model load).

Check your CUDA version with `nvidia-smi`, then install the matching build:

```bash
# CUDA 12.x (RTX 30xx / 40xx series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU is detected:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 5. Run the app

```bash
python app.py
```

Open your browser at **http://localhost:5000**

On first launch the model loads from `./saved_model` (~1 s on GPU, ~4 min on CPU).  
The first heatmap request also builds a cached eager-mode ViT twin (~1 s extra).

---

## Features

| Feature | Details |
|---|---|
| **Fine-tuned ViT** | `google/vit-base-patch16-224` — 98.52% accuracy on HAM10000 |
| **7-class prediction** | Melanoma, BCC, Actinic Keratoses, Benign Keratosis, Melanocytic Nevi, Dermatofibroma, Vascular Lesions |
| **Attention Rollout heatmap** | Real transformer attention weights (Abnar & Zuidema, 2020) |
| **Risk classification** | Malignant / Pre-Malignant / Benign with colour coding |
| **Low-confidence warning** | Amber banner when model confidence < 60% |
| **Recommendation engine** | Step-by-step clinical advice + Google Maps diagnostic centre link |
| **Auto image-type detection** | H&E stain heuristic — pathology vs external/dermatology |
| **Camera capture** | Live camera via `getUserMedia` |
| **Drag & drop upload** | PNG, JPG, BMP, TIFF, WEBP — max 16 MB |
| **Edge-case guards** | Rejects images < 32×32 px or unreadable files |
| **Animated CSS background** | Aurora gradient + floating orbs + dot grid (no video file needed) |
| **Responsive UI** | Desktop and mobile |

---

## Classes & Risk Levels

| Class | Category | Risk |
|---|---|---|
| Melanoma | Malignant | High |
| Basal Cell Carcinoma | Malignant | High |
| Actinic Keratoses | Pre-Malignant | Medium |
| Benign Keratosis-like Lesions | Benign | Low |
| Melanocytic Nevi | Benign | Low |
| Dermatofibroma | Benign | Low |
| Vascular Lesions | Benign | Low |

---

## Attention Rollout Heatmap

The saliency map uses **ViT Attention Rollout** rather than Grad-CAM (which requires CNN layers):

1. Run inference with `output_attentions=True` on an eager-mode model twin
2. Average attention over all 12 heads per layer
3. Add identity matrix (models the residual skip connection)
4. Row-normalise, then matrix-multiply through all 12 layers
5. Extract the `[CLS] → patches` row → reshape to 14×14 grid
6. Upsample to original image resolution, apply JET colormap, blend with original

The eager twin is cached on the model wrapper at first use — subsequent requests are instant.

---

## Notes

- This is a **research / educational tool**. AI predictions are NOT a medical diagnosis.
- For clinical deployment, additional validation on diverse populations is required.
- The `finetune.py` script documents the training procedure and hyperparameters.

---

## License

MIT — for educational and research purposes only.
