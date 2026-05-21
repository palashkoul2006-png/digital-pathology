"""
acoustic/export_model.py — Task B
===================================
Exports the trained CoughCNN to:
  1. TFLite (float16 quantized) → acoustic/saved_model/model.tflite
  2. TensorFlow.js LayersModel  → static/acoustic_model/

Usage (run AFTER train.py):
    python acoustic/export_model.py

Requirements:
    pip install tensorflowjs
"""

import os
import sys
import logging
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

SAVED_DIR   = os.path.join(os.path.dirname(__file__), 'saved_model')
TFLITE_PATH = os.path.join(SAVED_DIR, 'model.tflite')
KERAS_CKPT  = os.path.join(SAVED_DIR, 'best_model.h5')
TFJS_DIR    = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'static', 'acoustic_model'
)


def export_tflite(model):
    """Export float16 quantized TFLite model."""
    try:
        import tensorflow as tf
    except ImportError:
        logger.error("TensorFlow required: pip install tensorflow")
        sys.exit(1)

    logger.info("Converting to TFLite (float16 quantization) …")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(TFLITE_PATH) / 1024
    logger.info(f"✅ TFLite saved: {TFLITE_PATH}  ({size_kb:.1f} KB)")
    return TFLITE_PATH


def export_tfjs(saved_model_path: str):
    """Convert Keras SavedModel to TF.js LayersModel format."""
    try:
        import tensorflowjs as tfjs
    except ImportError:
        logger.error(
            "tensorflowjs required: pip install tensorflowjs\n"
            "Also ensure tensorflow is installed."
        )
        sys.exit(1)

    os.makedirs(TFJS_DIR, exist_ok=True)

    logger.info(f"Converting to TF.js LayersModel → {TFJS_DIR} …")
    tfjs.converters.save_keras_model(
        # Load from the .keras checkpoint for cleanest export
        __import__('tensorflow').keras.models.load_model(saved_model_path),
        TFJS_DIR
    )

    # List exported files
    files = os.listdir(TFJS_DIR)
    logger.info(f"✅ TF.js model files: {files}")
    logger.info(f"✅ Saved to: {TFJS_DIR}")


def verify_tflite(tflite_path: str):
    """Run a single inference with the TFLite interpreter to verify."""
    try:
        import tensorflow as tf
        import numpy as np
        from acoustic.preprocess import INPUT_SHAPE, CLASSES
    except ImportError:
        logger.warning("Skipping TFLite verification (import error).")
        return

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Dummy input
    dummy = np.zeros((1,) + INPUT_SHAPE, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    predicted = CLASSES[output.argmax()]
    logger.info(f"TFLite verification: output={output}  predicted='{predicted}'")
    logger.info("✅ TFLite inference OK")


def main():
    try:
        import tensorflow as tf
    except ImportError:
        logger.error("TensorFlow required: pip install tensorflow")
        sys.exit(1)

    # ── Load the trained model ─────────────────────────────────────────────
    keras_checkpoint = os.path.join(SAVED_DIR, 'best_model.h5')
    if os.path.exists(keras_checkpoint):
        logger.info(f"Loading best checkpoint: {keras_checkpoint}")
        model = tf.keras.models.load_model(keras_checkpoint)
    elif os.path.exists(SAVED_DIR):
        logger.info(f"Loading SavedModel: {SAVED_DIR}")
        model = tf.keras.models.load_model(SAVED_DIR)
    else:
        logger.error(
            f"No trained model found at {SAVED_DIR}.\n"
            "Run 'python acoustic/train.py' first."
        )
        sys.exit(1)

    logger.info(f"Model loaded. Parameters: {model.count_params():,}")

    # ── Export TFLite ──────────────────────────────────────────────────────
    export_tflite(model)
    verify_tflite(TFLITE_PATH)

    # ── Export TF.js ───────────────────────────────────────────────────────
    source = KERAS_CKPT if os.path.exists(KERAS_CKPT) else SAVED_DIR
    export_tfjs(source)

    logger.info("\n" + "=" * 60)
    logger.info("Export complete! Files ready for the web app:")
    logger.info(f"  TFLite : {TFLITE_PATH}")
    logger.info(f"  TF.js  : {TFJS_DIR}/")
    logger.info("=" * 60)
    logger.info("Now run: python app.py  then visit http://localhost:5000/acoustic")


if __name__ == '__main__':
    main()
