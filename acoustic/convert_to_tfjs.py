"""
acoustic/convert_to_tfjs.py
============================
Converts the trained Keras .h5 model directly to TF.js LayersModel format
WITHOUT using the tensorflowjs Python package (avoids all dependency conflicts).

The TF.js Layers format is:
  static/acoustic_model/model.json    — topology + weight manifest
  static/acoustic_model/weights.bin  — all weights concatenated as float32 LE

Usage:
    venv_acoustic\Scripts\python acoustic/convert_to_tfjs.py
"""

import os
import sys
import json
import struct
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SAVED_DIR   = os.path.join(os.path.dirname(__file__), 'saved_model')
H5_PATH     = os.path.join(SAVED_DIR, 'best_model.h5')
TFJS_OUT    = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'acoustic_model')

DTYPE_MAP = {
    'float32': 'float32',
    'float16': 'float16',
    'int32':   'int32',
}


def convert(h5_path: str, out_dir: str):
    try:
        import tensorflow as tf
    except ImportError:
        logger.error("TensorFlow required"); sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Loading model: {h5_path}")
    model = tf.keras.models.load_model(h5_path)
    logger.info(f"Model loaded — {model.count_params():,} parameters")

    # ── 1. Collect weights ─────────────────────────────────────────────────
    weight_specs = []
    raw_buffers  = []

    for layer in model.layers:
        for w in layer.weights:
            arr    = w.numpy().astype(np.float32)
            name   = w.name.replace(':', '/').replace('/', '_')
            flat   = arr.flatten()
            n_bytes = flat.nbytes

            weight_specs.append({
                'name':  name,
                'shape': list(arr.shape),
                'dtype': 'float32',
            })
            raw_buffers.append(flat.tobytes())

    # ── 2. Write weights.bin ───────────────────────────────────────────────
    bin_name = 'weights.bin'
    bin_path = os.path.join(out_dir, bin_name)
    total_bytes = 0
    weight_manifest_entries = []

    with open(bin_path, 'wb') as f:
        for spec, buf in zip(weight_specs, raw_buffers):
            byte_len = len(buf)
            f.write(buf)
            weight_manifest_entries.append({
                'name':  spec['name'],
                'shape': spec['shape'],
                'dtype': spec['dtype'],
            })
            total_bytes += byte_len

    logger.info(f"Weights binary: {total_bytes / 1024:.1f} KB → {bin_path}")

    # ── 3. Build model topology JSON (TF.js LayersModel format) ───────────
    model_config = json.loads(model.to_json())

    model_json = {
        'format':       'layers-model',
        'generatedBy':  'TensorFlow 2.x custom exporter (acoustic/convert_to_tfjs.py)',
        'convertedBy':  None,
        'modelTopology': {
            'class_name': model_config['class_name'],
            'config':     model_config['config'],
            'keras_version': tf.__version__,
            'backend':    'tensorflow',
        },
        'weightsManifest': [
            {
                'paths':   [bin_name],
                'weights': weight_manifest_entries,
            }
        ],
    }

    json_path = os.path.join(out_dir, 'model.json')
    with open(json_path, 'w') as f:
        json.dump(model_json, f, separators=(',', ':'))

    logger.info(f"Model JSON written → {json_path}")

    # ── 4. Quick sanity check ──────────────────────────────────────────────
    files = os.listdir(out_dir)
    json_kb  = os.path.getsize(json_path)  / 1024
    bin_kb   = os.path.getsize(bin_path)   / 1024
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"  model.json : {json_kb:.1f} KB")
    logger.info(f"  weights.bin: {bin_kb:.1f} KB")
    logger.info("\n✅ TF.js model ready! Files:")
    for f in files:
        logger.info(f"   {f}")
    logger.info("\nStart the Flask app and visit http://localhost:5000/acoustic")


if __name__ == '__main__':
    if not os.path.exists(H5_PATH):
        logger.error(f"No trained model at {H5_PATH}. Run acoustic/train.py first.")
        sys.exit(1)
    convert(H5_PATH, TFJS_OUT)
