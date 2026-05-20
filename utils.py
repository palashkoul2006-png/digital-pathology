"""
utils.py - Image pre-processing, type detection, and recommendation engine
for Digital Pathology Analysis.
"""

import os
import cv2
import numpy as np
import logging
import json
from google import genai

logger = logging.getLogger(__name__)

# ── Image pre-processing ──────────────────────────────────────────────────────

def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Load, resize, denoise, and normalize an image for model input.

    Returns
    -------
    np.ndarray of shape (1, 224, 224, 3), dtype float32, values in [0, 1].
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Gentle denoising to reduce sensor noise
    img = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10,
                                          templateWindowSize=7, searchWindowSize=21)

    # Resize with high-quality interpolation
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Add batch dimension
    return np.expand_dims(img, axis=0)


def detect_image_type(image_path: str) -> str:
    """
    Heuristically classify an image as 'pathology' (microscopic) or 'external'
    (skin / eye / ear / hair) based on colour statistics and texture entropy.

    Pathology images typically have:
      - High saturation variance (staining patterns)
      - High texture entropy (dense cellular structures)
      - Dominant purple/pink hues (H&E stain)

    Returns 'pathology' or 'external'.
    """
    img = cv2.imread(image_path)
    if img is None:
        return 'external'

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ── Hue analysis: H&E stain sits in pink/purple (hue ≈ 130–180 in HSV 0-180 scale)
    hue_channel = img_hsv[:, :, 0].astype(float)
    purple_pink_mask = ((hue_channel >= 130) & (hue_channel <= 180))
    purple_ratio = float(purple_pink_mask.sum()) / purple_pink_mask.size

    # ── Saturation variance (pathology images have high local saturation variance)
    sat_std = float(img_hsv[:, :, 1].std())

    # ── Texture entropy via grayscale histogram
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-8)
    entropy = -float(np.sum(hist_norm * np.log2(hist_norm + 1e-8)))

    logger.info(
        f"Image stats → purple_ratio={purple_ratio:.3f}, "
        f"sat_std={sat_std:.1f}, entropy={entropy:.2f}"
    )

    # Decision rule (tuned empirically; replace with a trained classifier if available)
    if purple_ratio > 0.12 or (sat_std > 55 and entropy > 6.5):
        return 'pathology'
    return 'external'


# ── Recommendation engine ─────────────────────────────────────────────────────

# ── Agentic AI Recommendation Engine (Gemini LLM) ─────────────────────────────

# Initialize the NEW Gemini Client
_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY environment variable is not set. "
        "Create a .env file with GEMINI_API_KEY=your_key_here and restart."
    )
client = genai.Client(api_key=_api_key)
def generate_agentic_report(
    prediction: str,
    confidence: float,
    risk_level: str,
    image_type: str,
) -> dict:
    """
    Takes the raw outputs from the ViT model and feeds them to the Gemini LLM 
    to generate a personalized, dynamic medical care plan.
    """
    
    system_prompt = f"""
    You are an expert dermatological AI assistant. A Vision Transformer AI has just analyzed a {image_type} image of a patient's skin lesion.
    
    Diagnostic Context:
    - AI Prediction: {prediction}
    - AI Confidence: {confidence}%
    - Risk Level: {risk_level}

    Your task is to provide a compassionate, factual, and strictly structured report. 
    DO NOT provide a definitive medical diagnosis. Always recommend seeing a doctor.

    Respond ONLY with a valid JSON object matching this exact structure:
    {{
        "medical_advice": "Clear, concise advice on what this prediction means and urgency of seeing a doctor.",
        "preventive_routine": ["Step 1", "Step 2", "Step 3"],
        "lifestyle_suggestions": ["Suggestion 1", "Suggestion 2"]
    }}
    """

    try:
        # Call the LLM
        response = client.models.generate_content(model='gemini-2.5-flash', contents=system_prompt)
        
        # Clean up the response to ensure it's pure JSON
        response_text = response.text.replace("```json", "").replace("```", "").strip()
        agent_report = json.loads(response_text)
        
    except Exception as e:
        logger.error(f"LLM Generation Failed: {e}")
        # Safe fallback if API fails
        agent_report = {
            "medical_advice": f"The AI detected {prediction} with {risk_level} risk. Please consult a dermatologist.",
            "preventive_routine": ["Monitor the lesion for changes in size or colour.", "Wear SPF 50+ daily."],
            "lifestyle_suggestions": ["Avoid direct sunlight during peak hours.", "Stay hydrated."]
        }

    # Format the dynamic LLM data to perfectly match what the React/JS frontend expects
    maps_link = (
        "https://www.google.com/maps/search/diagnostic+centre+near+me"
        if risk_level in ('High', 'Medium')
        else "https://www.google.com/maps/search/health+clinic+near+me"
    )

    image_context = "microscopic pathology image" if image_type == 'pathology' else "clinical/dermatological image"
    disclaimer = (
        f"This dynamic analysis was performed on a {image_context} using Agentic AI. "
        "Results are indicative only and do not constitute a medical diagnosis. "
        "Always consult a qualified medical professional."
    )

    # Combine routines for the frontend bullet points
    combined_steps = agent_report["preventive_routine"] + agent_report["lifestyle_suggestions"]

    return {
        'summary': agent_report["medical_advice"],
        'steps': combined_steps,
        'maps_link': maps_link,
        'disclaimer': disclaimer,
    }


def chat_with_agent(user_message: str, context_summary: str, chat_history: str) -> str:
    system_prompt = f"""
    You are an empathetic, expert dermatological AI assistant. 
    PATIENT'S DIAGNOSTIC CONTEXT: \n{context_summary}
    PREVIOUS CONVERSATION HISTORY: \n{chat_history if chat_history else "None."}
    PATIENT'S NEW QUESTION: \n{user_message}
    Provide a concise, helpful, and safe answer in plain text. Explain medical jargon simply. 
    Never provide a definitive medical diagnosis. Always recommend consulting a doctor.
    """
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=system_prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Chat API Failed: {e}")
        return "I'm having trouble connecting right now. Please consult your doctor."