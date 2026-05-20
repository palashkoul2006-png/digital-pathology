"""
models.py - AI model management for Digital Pathology Analysis
Loads the fine-tuned ViT (google/vit-base-patch16-224) from ./saved_model
and maps 7-class skin-cancer probabilities to clinical categories.
"""

import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Class metadata ────────────────────────────────────────────────────────────
# The 7 HAM10000 classes ordered as they appear in the saved model's id2label
CLASSES = [
    'actinic_keratoses',
    'basal_cell_carcinoma',
    'benign_keratosis-like_lesions',
    'dermatofibroma',
    'melanocytic_Nevi',
    'melanoma',
    'vascular_lesions',
]

# Clinical risk for each class
_CLASS_RISK = {
    'actinic_keratoses':           ('Pre-Malignant', 'Medium'),
    'basal_cell_carcinoma':        ('Malignant',     'High'),
    'benign_keratosis-like_lesions': ('Benign',      'Low'),
    'dermatofibroma':              ('Benign',         'Low'),
    'melanocytic_Nevi':            ('Benign',         'Low'),
    'melanoma':                    ('Malignant',      'High'),
    'vascular_lesions':            ('Benign',         'Low'),
}

# Human-readable display names
_DISPLAY_NAMES = {
    'actinic_keratoses':           'Actinic Keratoses',
    'basal_cell_carcinoma':        'Basal Cell Carcinoma',
    'benign_keratosis-like_lesions': 'Benign Keratosis',
    'dermatofibroma':              'Dermatofibroma',
    'melanocytic_Nevi':            'Melanocytic Nevi',
    'melanoma':                    'Melanoma',
    'vascular_lesions':            'Vascular Lesions',
}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models() -> dict:
    """
    Load the fine-tuned ViT from ./saved_model.
    Returns a dict with a single 'vit' key for compatibility with app.py.
    Falls back to a demo stub if the model directory is missing.
    """
    try:
        from transformers import ViTForImageClassification, ViTImageProcessor
        import torch

        model_path = './saved_model'
        logger.info(f"Loading fine-tuned ViT from {model_path} …")

        processor = ViTImageProcessor.from_pretrained(model_path)
        model     = ViTForImageClassification.from_pretrained(model_path)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()
        logger.info(f"ViT loaded on {device.upper()} ✅")

        wrapper = _ViTWrapper(model, processor, device)
        # Return both keys so app.py's model_key logic still works
        return {'vit': wrapper, 'mobilenet': wrapper, 'resnet': wrapper}

    except Exception as exc:
        logger.warning(f"Could not load ViT model ({exc}). Running in demo mode.")
        stub = _DemoModel()
        return {'vit': stub, 'mobilenet': stub, 'resnet': stub}


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_image(model, preprocessed_array: np.ndarray, image_type: str, image_path: str = None) -> dict:
    """
    Run inference and return clinical prediction details.

    Parameters
    ----------
    model              : _ViTWrapper or _DemoModel
    preprocessed_array : (1, 224, 224, 3) float32 array  [0-1 range]
    image_type         : 'external' | 'pathology'  (kept for API compatibility)

    Returns
    -------
    dict with keys: prediction, display_name, confidence, risk_level,
                    category, all_probs
    """
    return model.predict(preprocessed_array, image_path=image_path)


# ── ViT wrapper ───────────────────────────────────────────────────────────────

class _ViTWrapper:
    """Wraps a HuggingFace ViT model for inference from a numpy array."""

    def __init__(self, model, processor, device: str):
        self.model     = model
        self.processor = processor
        self.device    = device
        # Expose a .layers attribute so heatmaps.py's TF check stays False
        self.layers    = []

    def predict(self, arr: np.ndarray, image_path: str = None) -> dict:
        """
        Run ViT inference.
        Bypasses ViTImageProcessor to avoid torchvision dependency —
        normalises manually with ViT's standard mean=0.5 / std=0.5.
        If image_path is provided the original file is loaded directly
        (avoids the double-preprocessing chain through OpenCV).
        """
        import torch

        # ── Load / prepare PIL image ──────────────────────────────────
        if image_path:
            # Use original file for best quality
            pil_img = Image.open(image_path).convert('RGB')
            pil_img = pil_img.resize((224, 224), Image.LANCZOS)
        else:
            # Reconstruct from pre-processed numpy array (1, 224, 224, 3)
            img_np  = (arr[0] * 255.0).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, mode='RGB')

        # ── Manual ViT normalisation (mean=0.5, std=0.5 per channel) ─
        img_array = np.array(pil_img, dtype=np.float32) / 255.0   # [0,1]
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        img_array = (img_array - mean) / std                       # [-1,1]

        # HWC → CHW → batch
        img_tensor = torch.tensor(
            img_array.transpose(2, 0, 1),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)                             # (1,3,224,224)

        # ── Inference ─────────────────────────────────────────────────
        with torch.no_grad():
            logits = self.model(pixel_values=img_tensor).logits   # (1, 7)
            probs  = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        logger.info(
            f"ViT probs: "
            + ", ".join(f"{CLASSES[i]}={probs[i]:.3f}" for i in range(len(CLASSES)))
        )

        top_idx   = int(np.argmax(probs))
        top_class = CLASSES[top_idx]
        top_conf  = float(probs[top_idx])
        category, risk_level = _CLASS_RISK[top_class]

        return {
            'prediction':   top_class,
            'display_name': _DISPLAY_NAMES[top_class],
            'confidence':   top_conf,
            'risk_level':   risk_level,
            'category':     category,
            'all_probs':    {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
        }


# ── Demo stub ─────────────────────────────────────────────────────────────────

class _DemoModel:
    """Deterministic demo when the real model is unavailable."""

    def __init__(self):
        self.layers = []

    def predict(self, arr: np.ndarray, image_path: str = None) -> dict:
        mean_val  = float(np.mean(arr))
        idx       = int(mean_val * len(CLASSES)) % len(CLASSES)
        top_class = CLASSES[idx]
        confidence = round(0.70 + abs(mean_val - 0.45) * 0.4, 4)
        confidence = min(confidence, 0.97)
        category, risk_level = _CLASS_RISK[top_class]

        return {
            'prediction':   top_class,
            'display_name': _DISPLAY_NAMES[top_class],
            'confidence':   confidence,
            'risk_level':   risk_level,
            'category':     category,
            'all_probs':    {c: 1/len(CLASSES) for c in CLASSES},
        }
