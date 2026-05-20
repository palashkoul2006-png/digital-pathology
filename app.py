"""
app.py - Main Flask application for Digital Pathology Image Analysis
Handles routing, image upload, model inference, and result delivery.
Now uses the fine-tuned ViT (98.52% accuracy) for 7-class skin cancer prediction.
"""

import os
import uuid
import base64
import logging
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from datetime import datetime

# Load environment variables from .env file (before any other imports that need them)
load_dotenv()

from models import load_models, predict_image
from utils import preprocess_image, detect_image_type, generate_agentic_report, chat_with_agent
from heatmaps import generate_heatmap
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024          # 16 MB limit
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── Database Setup (MongoDB) ────────────────────────────────────────────────
try:
    # Connect to your local MongoDB (Ensure MongoDB is running locally!)
    mongo_client = MongoClient('mongodb://localhost:27017/')
    db = mongo_client['pathai_db']
    patients_collection = db['patients']
    logger.info("MongoDB connected successfully.")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")

# ── Load models at startup ────────────────────────────────────────────────────
logger.info("Loading AI models...")
models = load_models()
logger.info("Models loaded successfully.")
# ── Load Generative Engine (Stable Diffusion + LoRA) ──────────────────────────
logger.info("Loading Generative Engine...")
try:
    gen_pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        r"C:\AI_Training\base_model\v1-5-pruned-emaonly.safetensors",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    gen_pipe.load_lora_weights(r"C:\Users\palas\OneDrive - Vishwakarma Institute of Technology\Desktop\digital_pathology\PathAI_LoRA\model\pathai_melanoma_v1.safetensors")
    
    # CRITICAL VRAM SAVER FOR RTX 3050 6GB:
    gen_pipe.enable_model_cpu_offload() 
    logger.info("Generative Engine loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Generative Engine: {e}")
    gen_pipe = None


def allowed_file(filename: str) -> bool:
    """Check if file extension is permitted."""
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive an uploaded image (multipart or base64), run inference,
    and return a JSON result with prediction, confidence, risk, heatmap,
    class probabilities, and recommendation.
    """
    try:
        image_data = None
        # Extract Patient ID (default to Anonymous if none is provided)
        patient_id = request.form.get('patient_id', 'Anonymous') if 'file' in request.files else request.json.get('patient_id', 'Anonymous')

        # ── Accept multipart file upload ──────────────────────────────────────
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected.'}), 400
            if not allowed_file(file.filename):
                return jsonify({'error': 'Unsupported file type.'}), 400

            filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_data = filepath

        # ── Accept base64-encoded image (camera capture) ─────────────────────
        elif request.is_json and 'image_b64' in request.json:
            b64_str = request.json['image_b64']
            if ',' in b64_str:
                b64_str = b64_str.split(',', 1)[1]
            raw = base64.b64decode(b64_str)
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, 'wb') as f:
                f.write(raw)
            image_data = filepath

        else:
            return jsonify({'error': 'No image provided.'}), 400

        # ── Pre-process & detect image type ───────────────────────────────────
        # Sanity-check: reject images that are too small or corrupt
        try:
            from PIL import Image as _PIL
            with _PIL.open(image_data) as _img:
                _w, _h = _img.size
            if _w < 32 or _h < 32:
                return jsonify({
                    'error': f'Image too small ({_w}×{_h} px). Please upload at least 32×32 px.'
                }), 400
        except Exception as _e:
            return jsonify({'error': f'Could not read image: {str(_e)}'}), 400

        preprocessed = preprocess_image(image_data)
        image_type   = detect_image_type(image_data)
        logger.info(f"Image type detected: {image_type}")

        # ── Run inference with fine-tuned ViT ─────────────────────────────────
        # Use 'vit' key; falls back to mobilenet/resnet aliases if needed
        model_key = 'vit'
        # 1. Pass 'image_data' (the file path) instead of 'preprocessed'
        result    = predict_image(models[model_key], preprocessed, image_type, image_path=image_data)

        # ── Generate attention / saliency heatmap ─────────────────────────────
        # 2. Pass 'None' for the array, and 'image_data' for the path
        heatmap_b64 = generate_heatmap(models[model_key], None, image_data)

        # ── Call the Agentic LLM ───────────────────────────────────────────────
        recommendation = generate_agentic_report(
            prediction=result['display_name'],
            confidence=round(result['confidence'] * 100, 2),
            risk_level=result['risk_level'],
            image_type=image_type
        )

        # ── Save to Database (MongoDB) ────────────────────────────────────────
        try:
            new_record = {
                "patient_id": patient_id,
                "date": datetime.utcnow(),
                "image_path": image_data,
                "prediction": result['display_name'],
                "confidence": round(result['confidence'] * 100, 2),
                "risk_level": result['risk_level']
            }
            patients_collection.insert_one(new_record)
        except Exception as db_err:
            logger.error(f"Failed to save record to MongoDB: {db_err}")

        # ── Encode original image for preview ─────────────────────────────────
        with open(image_data, 'rb') as img_file:
            original_b64 = base64.b64encode(img_file.read()).decode('utf-8')

        return jsonify({
            'status':         'success',
            'image_type':     image_type,
            'model_used':     'ViT (fine-tuned)',
            # Specific 7-class result
            'prediction':     result['prediction'],
            'display_name':   result['display_name'],
            'category':       result['category'],
            # Confidence and risk
            'confidence':     round(result['confidence'] * 100, 2),
            'risk_level':     result['risk_level'],
            # Per-class probabilities for bar chart
            'all_probs':      {k: round(v * 100, 2) for k, v in result['all_probs'].items()},
            # Visual outputs
            'recommendation': recommendation,
            'heatmap':        heatmap_b64,
            'original_image': f"data:image/jpeg;base64,{original_b64}",
        })

    except Exception as exc:
        logger.exception("Prediction error")
        return jsonify({'error': f'Analysis failed: {str(exc)}'}), 500

@app.route('/api/simulate_progression', methods=['POST'])
def simulate_progression():
    """
    Takes a benign image and uses the custom Stable Diffusion LoRA 
    to simulate how a p4th4i melanoma might develop on that specific skin structure.
    """
    if gen_pipe is None:
        return jsonify({'error': 'Generative engine is offline.'}), 503

    try:
        # 1. Catch the uploaded image
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected.'}), 400
            
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file.'}), 400

        # 2. Format the image for Stable Diffusion (must be 512x512)
        init_image = Image.open(file.stream).convert("RGB")
        init_image = init_image.resize((512, 512))

        # 3. The Clinical Prompt & Generation
        prompt = "macro clinical photography of a p4th4i melanoma on skin, highly detailed, sharp focus, medical imaging"
        
        # 'strength' of 0.65 keeps the original skin structure but adds severe melanoma textures
        generated_image = gen_pipe(
            prompt=prompt, 
            image=init_image, 
            strength=0.65, 
            guidance_scale=7.5
        ).images[0]

        # 4. Save the synthetic image
        filename = secure_filename(f"simulated_{uuid.uuid4()}.jpg")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        generated_image.save(filepath)

        # 5. Encode to base64 to send straight to the React frontend
        with open(filepath, 'rb') as img_file:
            synthetic_b64 = base64.b64encode(img_file.read()).decode('utf-8')

        return jsonify({
            'status': 'success',
            'message': 'Progression simulated successfully',
            'synthetic_image': f"data:image/jpeg;base64,{synthetic_b64}"
        })

    except Exception as exc:
        logger.exception("Simulation error")
        return jsonify({'error': f'Simulation failed: {str(exc)}'}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    context_summary = data.get('context', 'No specific context provided.')
    chat_history = data.get('history', '')

    if not user_message:
        return jsonify({'error': 'Message cannot be empty'}), 400

    try:
        ai_reply = chat_with_agent(user_message, context_summary, chat_history)
        return jsonify({'status': 'success', 'reply': ai_reply})
    except Exception as e:
        logger.error(f"Chat Route Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<patient_id>', methods=['GET'])
def get_history(patient_id):
    """Fetches chronological analysis history for a specific patient from MongoDB."""
    try:
        # Query MongoDB for the patient and sort by date ascending (1)
        records = patients_collection.find({"patient_id": patient_id}).sort("date", 1)
        
        history = []
        for r in records:
            history.append({
                'date': r['date'].strftime("%b %d, %Y - %H:%M"),
                'prediction': r['prediction'],
                'confidence': r['confidence'],
                'risk_level': r['risk_level']
            })
            
        return jsonify({'status': 'success', 'history': history})
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        return jsonify({'error': str(e)}), 500

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
