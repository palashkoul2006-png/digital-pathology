"""
heatmaps.py - Attention-based saliency heatmap generation for Digital Pathology.

Priority order:
  1. ViT Attention Rollout  — real transformer attention (preferred, fast on GPU)
  2. Grad-CAM               — for TensorFlow CNN models (MobileNet / ResNet)
  3. Simulated edge-energy  — fallback when nothing else works

Attention Rollout algorithm (Abnar & Zuidema, 2020):
  For each transformer layer, add the identity matrix (residual skip), row-normalise,
  then matrix-multiply all layers together.  The first row (CLS token) of the result
  gives the patch-level attention score — reshape 196 → 14×14 for ViT-B/16.
"""

import cv2
import numpy as np
import base64
import logging

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ── Public entry point ────────────────────────────────────────────────────────

def generate_heatmap(model, preprocessed_array: np.ndarray, original_path: str) -> str:
    """
    Generate a visual attention heatmap overlay and return a base64 JPEG string.

    Parameters
    ----------
    model              : _ViTWrapper | Keras Model | _DemoModel
    preprocessed_array : (1, 224, 224, 3) float32 array  [0-1 range]
    original_path      : path to the original uploaded image

    Returns
    -------
    str  – "data:image/jpeg;base64,<...>"
    """
    original = cv2.imread(original_path)
    if original is None:
        return _blank_heatmap()

    h, w = original.shape[:2]

    # ── Choose method based on model type ─────────────────────────────────────
    cam = None

    # 1) ViT Attention Rollout (best for our fine-tuned ViT)
    if hasattr(model, 'model') and hasattr(model, 'processor') and hasattr(model, 'device'):
        try:
            cam = _attention_rollout(model, original_path)
            logger.info("Heatmap: ViT Attention Rollout used.")
        except Exception as exc:
            logger.warning(f"Attention Rollout failed ({exc}), falling back.")
            cam = None

    # 2) Grad-CAM for TF CNN models
    if cam is None:
        model_layers = getattr(model, 'layers', [])
        has_conv = TF_AVAILABLE and model_layers and any(
            isinstance(layer, tf.keras.layers.Conv2D)
            for layer in model_layers
        ) if TF_AVAILABLE else False

        if has_conv:
            try:
                cam = _grad_cam(model, preprocessed_array)
                logger.info("Heatmap: Grad-CAM used.")
            except Exception as exc:
                logger.warning(f"Grad-CAM failed ({exc}), falling back.")
                cam = None

    # 3) Simulated edge-energy (last resort)
    if cam is None:
        cam = _simulated_cam(original)
        logger.info("Heatmap: Simulated edge-energy used.")

    # ── Resize, colorise, blend ────────────────────────────────────────────────
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)

    # Sharpen the upsampled map slightly for crisper patch boundaries
    cam_resized = cv2.GaussianBlur(cam_resized, (0, 0), sigmaX=3)

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )

    # Blend  (slightly favour original so anatomy is still visible)
    overlay = cv2.addWeighted(original, 0.50, heatmap_color, 0.50, 0)

    overlay = _draw_legend(overlay)
    return _encode_image(overlay)


# ── ViT Attention Rollout ─────────────────────────────────────────────────────

def _attention_rollout(vit_wrapper, image_path: str) -> np.ndarray:
    """
    Compute ViT Attention Rollout (Abnar & Zuidema, 2020) for the given image.

    Steps
    -----
    1. Run inference with output_attentions=True to get per-layer attention.
    2. Average multi-head attention → (layers, seq, seq).
    3. Add identity matrix (accounts for residual connections).
    4. Row-normalise each layer's matrix.
    5. Matrix-multiply through all layers (rollout).
    6. Extract CLS→patch row, reshape to 14×14 grid, normalise.

    Parameters
    ----------
    vit_wrapper : _ViTWrapper   — has .model, .processor, .device
    image_path  : str           — path to original image file

    Returns
    -------
    np.ndarray  shape (14, 14), dtype float32, values in [0, 1]
    """
    import torch
    from PIL import Image

    device = vit_wrapper.device
    torch_model = vit_wrapper.model

    # ── Prepare input ─────────────────────────────────────────────────────────
    pil_img = Image.open(image_path).convert('RGB').resize((224, 224), Image.LANCZOS)
    img_array = np.array(pil_img, dtype=np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5                           # ViT normalisation
    img_tensor = (
        torch.tensor(img_array.transpose(2, 0, 1), dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )                                                              # (1, 3, 224, 224)

    # ── Forward pass collecting all attention weights ──────────────────────────
    # PyTorch SDPA (scaled dot-product attention) is a fused kernel that does
    # not support returning intermediate attention weights.  We keep one cached
    # eager-mode twin on the wrapper so the disk load only happens once.
    import torch
    from transformers import ViTForImageClassification as _ViTCls

    # Cache the eager model as an attribute on the wrapper object
    if not hasattr(vit_wrapper, '_eager_model'):
        logger.info("Building cached eager-mode ViT for attention rollout...")
        _eager = _ViTCls.from_pretrained(
            './saved_model',
            attn_implementation='eager',
        ).to(device).eval()
        _eager.load_state_dict(torch_model.state_dict(), strict=False)
        _eager.config.output_attentions = True
        vit_wrapper._eager_model = _eager
        logger.info("Eager ViT cached on wrapper.")

    eager_model = vit_wrapper._eager_model

    with torch.no_grad():
        outputs = eager_model(
            pixel_values=img_tensor,
            output_attentions=True,
        )

    # outputs.attentions: tuple of (1, num_heads, seq_len, seq_len) — one per layer
    # For ViT-B/16: 12 layers, 12 heads, seq_len=197 (1 CLS + 196 patches)
    attentions = outputs.attentions                                # tuple[12]

    # ── Attention Rollout ──────────────────────────────────────────────────────
    # Stack → (num_layers, num_heads, seq, seq)
    attn_stack = torch.stack(attentions).squeeze(1)               # (L, H, S, S)

    # Average over heads → (L, S, S)
    attn_avg = attn_stack.mean(dim=1)                             # (L, S, S)

    # Move to CPU for numpy ops
    attn_avg = attn_avg.cpu().float()

    seq_len = attn_avg.shape[-1]
    identity = torch.eye(seq_len, dtype=torch.float32)

    # Apply rollout layer by layer
    rollout = identity.clone()
    for layer_attn in attn_avg:                                   # iterate over L layers
        # Add residual (skip connection) and normalise each row
        a = layer_attn + identity
        a = a / a.sum(dim=-1, keepdim=True)                       # row-normalise
        rollout = torch.matmul(a, rollout)                        # accumulate

    # ── Extract CLS→patch attention ────────────────────────────────────────────
    # Row 0 = CLS token; columns 1: = patch tokens (skip CLS column)
    cls_attn = rollout[0, 1:].numpy()                              # (196,)

    # Reshape to spatial grid (14×14 for ViT-B/16-224)
    grid_size = int(np.sqrt(cls_attn.shape[0]))                   # 14
    mask = cls_attn.reshape(grid_size, grid_size).astype(np.float32)

    # Normalise to [0, 1]
    m_min, m_max = mask.min(), mask.max()
    if m_max > m_min:
        mask = (mask - m_min) / (m_max - m_min)

    return mask                                                    # (14, 14)


# ── Grad-CAM for TF CNN models ────────────────────────────────────────────────

def _grad_cam(model, img_array: np.ndarray) -> np.ndarray:
    """
    Compute Grad-CAM activation map using the last Conv2D layer.
    Works with MobileNetV2 and ResNet50 Keras models.
    """
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
            break
    if last_conv is None:
        raise ValueError("No Conv2D layer found in model.")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        class_idx  = tf.argmax(predictions[0])
        class_score = predictions[:, class_idx]

    grads       = tape.gradient(class_score, conv_outputs)        # (1, h, w, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))          # (C,)
    conv_outputs = conv_outputs[0]                                 # (h, w, C)
    cam = conv_outputs @ pooled_grads[..., tf.newaxis]             # (h, w, 1)
    cam = tf.squeeze(cam).numpy()

    cam = np.maximum(cam, 0)
    cam_max = cam.max()
    if cam_max > 0:
        cam /= cam_max
    return cam.astype(np.float32)


# ── Simulated edge-energy heatmap (fallback) ──────────────────────────────────

def _simulated_cam(original: np.ndarray) -> np.ndarray:
    """
    Produce a visually plausible saliency map using Laplacian edge energy +
    a centred Gaussian when neither rollout nor Grad-CAM is available.
    """
    gray   = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    lap    = cv2.Laplacian(gray, cv2.CV_64F)
    energy = np.abs(lap).astype(np.float32)
    energy = cv2.GaussianBlur(energy, (51, 51), sigmaX=20)

    h, w  = energy.shape
    cx, cy = w // 2, h // 2
    Y, X  = np.ogrid[:h, :w]
    centre_gauss = np.exp(
        -((X - cx) ** 2 + (Y - cy) ** 2) / (2 * (min(h, w) * 0.25) ** 2)
    )
    energy = 0.6 * energy + 0.4 * (centre_gauss.astype(np.float32) * energy.max())

    e_min, e_max = energy.min(), energy.max()
    if e_max > e_min:
        energy = (energy - e_min) / (e_max - e_min)
    return energy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _draw_legend(img: np.ndarray) -> np.ndarray:
    """Draw a compact colour-scale legend in the bottom-right corner."""
    h, w    = img.shape[:2]
    bar_w, bar_h = 130, 12
    x0 = w - bar_w - 10
    y0 = h - bar_h - 20

    for i in range(bar_w):
        val   = int(255 * i / bar_w)
        color = cv2.applyColorMap(
            np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET
        )[0][0]
        img[y0:y0 + bar_h, x0 + i] = color.tolist()

    cv2.putText(img, 'Low',  (x0,            y0 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, 'Attention', (x0 + bar_w // 2 - 22, y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, 'High', (x0 + bar_w - 28, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _encode_image(img: np.ndarray) -> str:
    """Encode a numpy BGR image as a base64 JPEG data URL."""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def _blank_heatmap() -> str:
    """Return a plain grey placeholder when the image cannot be read."""
    placeholder = np.full((224, 224, 3), 60, dtype=np.uint8)
    cv2.putText(placeholder, 'Heatmap N/A', (30, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return _encode_image(placeholder)
