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
import joblib
import numpy as np

# Load environment variables from .env file (before any other imports that need them)
load_dotenv(override=True)

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

# --- LUNA MODEL SETUP ---
LUNA_MODEL_PATH = "model.pkl"
try:
    luna_model = joblib.load(LUNA_MODEL_PATH)
    logger.info("✅ Luna Cycle Health model loaded successfully.")
except FileNotFoundError:
    luna_model = None
    logger.warning("⚠️ Luna model.pkl not found. Make sure it is in the root directory.")
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
        record_id = None
        try:
            new_record = {
                "patient_id": patient_id,
                "date": datetime.now(),
                "image_path": image_data,
                "prediction": result['display_name'],
                "confidence": round(result['confidence'] * 100, 2),
                "risk_level": result['risk_level']
            }
            insert_res = patients_collection.insert_one(new_record)
            record_id = str(insert_res.inserted_id)
        except Exception as db_err:
            logger.error(f"Failed to save record to MongoDB: {db_err}")

        # ── Encode original image for preview ─────────────────────────────────
        with open(image_data, 'rb') as img_file:
            original_b64 = base64.b64encode(img_file.read()).decode('utf-8')

        return jsonify({
            'status':         'success',
            'record_id':      record_id,
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
        query = {"patient_id": patient_id}
        
        # Exclude the current record if provided
        exclude_id = request.args.get('exclude')
        if exclude_id:
            from bson.objectid import ObjectId
            try:
                query["_id"] = {"$ne": ObjectId(exclude_id)}
            except Exception:
                pass # Ignore invalid ObjectIds

        # Query MongoDB for the patient and sort by date ascending (1)
        records = patients_collection.find(query).sort("date", 1)
        
        history = []
        for r in records:
            # Return as ISO format for the frontend
            iso_date = r['date'].isoformat()
                
            history.append({
                'date': iso_date,
                'prediction': r['prediction'],
                'confidence': r['confidence'],
                'risk_level': r['risk_level']
            })
            
        return jsonify({'status': 'success', 'history': history})
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        return jsonify({'error': str(e)}), 500

# ── Acoustic Stethoscope ──────────────────────────────────────────────────────
@app.route('/acoustic')
def acoustic():
    """Serve the Cough-to-Clinic Acoustic Virtual Stethoscope page."""
    return render_template('acoustic.html')


@app.route('/acoustic/analyze', methods=['POST'])
def acoustic_analyze():
    """
    Accept a recorded audio blob from the browser and pass it to a standalone
    inference script running in the venv_acoustic environment. This avoids
    TensorFlow/Keras dependency conflicts in the main Flask environment.
    """
    import tempfile, subprocess, json

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    try:
        suffix = '.webm'
        content_type = audio_file.content_type or ''
        if 'ogg' in content_type:
            suffix = '.ogg'
        elif 'wav' in content_type:
            suffix = '.wav'
        elif 'mp4' in content_type or 'mp4a' in content_type:
            suffix = '.mp4'

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        # ── Shell out to the venv_acoustic Python environment ─────────────────
        # Ensure we use the isolated environment that has tensorflow installed correctly
        python_exe = os.path.join(os.path.dirname(__file__), 'venv_acoustic', 'Scripts', 'python.exe')
        infer_script = os.path.join(os.path.dirname(__file__), 'acoustic', 'infer.py')
        
        if not os.path.exists(python_exe):
            # Fallback for linux/mac if someone runs this elsewhere
            python_exe = os.path.join(os.path.dirname(__file__), 'venv_acoustic', 'bin', 'python')

        # Run inference and capture stdout (JSON string)
        result = subprocess.run([python_exe, infer_script, tmp_path], 
                                capture_output=True, text=True)
        
        # Clean up temp file
        os.unlink(tmp_path)

        if result.returncode != 0:
            logger.error(f"Inference script failed: STDERR={result.stderr} STDOUT={result.stdout}")
            return jsonify({'error': f'Inference failed (exit code {result.returncode}). See server logs for details.'}), 500
            
        # The script prints a JSON string to stdout
        try:
            # If the script prints any TF warnings, grab only the last line (the JSON)
            lines = [line for line in result.stdout.strip().split('\n') if line.startswith('{')]
            output_json = lines[-1] if lines else result.stdout.strip()
            
            parsed = json.loads(output_json)
            
            if 'error' in parsed:
                return jsonify({'error': parsed['error']}), 500
                
            return jsonify(parsed)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse inference output: {result.stdout}")
            return jsonify({'error': f'Invalid output from inference: {e}'}), 500

    except Exception as e:
        logger.error(f"Acoustic analysis error: {e}", exc_info=True)
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass
        return jsonify({'error': str(e)}), 500


# ── LUNA ROUTES ───────────────────────────────────────────────────────────────
@app.route('/luna')
def luna_page():
    # Renders the Luna application UI
    return render_template('luna_app.html')

@app.route('/luna_landing')
def luna_landing_page():
    # Renders the Luna landing page
    return render_template('luna_landing.html')

@app.route("/predict_luna", methods=["POST"])
def predict_luna():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        required_fields = ["age", "bmi", "cycleLen", "periodLen", "stress", "sleep", "exercise"]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Extract features
        age = float(data["age"])
        bmi = float(data["bmi"])
        cycleLen = float(data["cycleLen"])
        periodLen = float(data["periodLen"])
        stress = float(data["stress"])
        sleep = float(data["sleep"])

        # Convert exercise string to numeric
        exercise_map = {"Low": 1, "Moderate": 2, "High": 3}
        exercise_val = exercise_map.get(data["exercise"], 2)

        features = np.array([[age, bmi, stress, exercise_val, sleep, cycleLen, periodLen]])

        if luna_model is None:
            return jsonify({"error": "Luna model is not loaded on the server."}), 500

        prediction = luna_model.predict(features)[0]

        return jsonify({
            "status": "success",
            "prediction": int(prediction)
        })

    except Exception as e:
        logger.error(f"Luna Error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

