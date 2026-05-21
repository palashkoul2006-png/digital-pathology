"""
acoustic/convert_to_tfjs.py
============================
Converts the trained Keras .h5 model directly to TF.js LayersModel format
WITHOUT using the tensorflowjs Python package.

This version hand-crafts the TF.js topology JSON to be fully compatible
with TF.js 3.x/4.x, sidestepping all Keras 3 format incompatibilities:
  - batch_shape  → batchInputShape
  - inbound_nodes as objects → inbound_nodes as arrays-of-arrays
  - module/registered_name fields stripped out
  - dtype objects → plain 'float32' strings

Usage:
    venv_acoustic\\Scripts\\python acoustic/convert_to_tfjs.py
"""

import os
import sys
import json
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SAVED_DIR = os.path.join(os.path.dirname(__file__), 'saved_model')
H5_PATH   = os.path.join(SAVED_DIR, 'best_model.h5')
TFJS_OUT  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'acoustic_model')


# ── Helpers to clean config dicts for TF.js ─────────────────────────────────

def clean_dtype(v):
    """If v is a Keras 3 DTypePolicy dict, return plain string. Else return v."""
    if isinstance(v, dict) and v.get('class_name') == 'DTypePolicy':
        return v.get('config', {}).get('name', 'float32')
    return v


def clean_initializer(d):
    """Strip Keras 3 module/registered_name from initializer configs."""
    if not isinstance(d, dict):
        return d
    out = {'class_name': d.get('class_name', 'GlorotUniform'), 'config': d.get('config', {})}
    return out


def clean_layer_config(cfg: dict) -> dict:
    """
    Strip Keras 3 extras and normalise a layer config dict to what TF.js expects.
    """
    SKIP = {'module', 'registered_name', 'quantization_config'}
    out = {}
    for k, v in cfg.items():
        if k in SKIP:
            continue
        if k == 'dtype':
            out[k] = clean_dtype(v)
        elif k in ('kernel_initializer', 'bias_initializer',
                   'gamma_initializer', 'beta_initializer',
                   'moving_mean_initializer', 'moving_variance_initializer',
                   'recurrent_initializer', 'embeddings_initializer'):
            out[k] = clean_initializer(v)
        elif k in ('kernel_regularizer', 'bias_regularizer', 'activity_regularizer',
                   'gamma_regularizer', 'beta_regularizer',
                   'kernel_constraint', 'bias_constraint',
                   'gamma_constraint', 'beta_constraint',
                   'recurrent_regularizer', 'recurrent_constraint'):
            out[k] = v  # keep as-is (None is fine)
        else:
            out[k] = v
    return out


def convert_inbound_nodes(keras3_inbound):
    """
    Keras 3 inbound_nodes format:
      [{"args": [{"class_name": "__keras_tensor__", "config": {"keras_history": [layerName, nodeIdx, tensorIdx]}}], "kwargs": {...}}]

    TF.js expected format:
      [[[layerName, nodeIdx, tensorIdx, {}]]]
    """
    if not keras3_inbound:
        return []

    tfjs_inbound = []
    for node in keras3_inbound:
        args = node.get('args', [])
        connections = []
        for arg in args:
            if isinstance(arg, dict) and arg.get('class_name') == '__keras_tensor__':
                hist = arg['config']['keras_history']
                layer_name, node_idx, tensor_idx = hist[0], hist[1], hist[2]
                connections.append([layer_name, node_idx, tensor_idx, {}])
            elif isinstance(arg, list):
                # Multiple inputs
                for a in arg:
                    if isinstance(a, dict) and a.get('class_name') == '__keras_tensor__':
                        hist = a['config']['keras_history']
                        connections.append([hist[0], hist[1], hist[2], {}])
        if connections:
            tfjs_inbound.append(connections)

    return tfjs_inbound


def build_tfjs_layer(layer_cfg: dict) -> dict:
    """Convert a single Keras 3 layer config entry into a TF.js compatible dict."""
    class_name    = layer_cfg.get('class_name', '')
    name          = layer_cfg.get('name', '')
    config        = layer_cfg.get('config', {})
    inbound_nodes = layer_cfg.get('inbound_nodes', [])
    build_config  = layer_cfg.get('build_config', {})

    # Clean the per-layer config
    tfjs_config = clean_layer_config(config)

    # InputLayer: rename batch_shape → batchInputShape
    if class_name == 'InputLayer':
        if 'batch_shape' in tfjs_config:
            tfjs_config['batchInputShape'] = tfjs_config.pop('batch_shape')
        # Remove fields TF.js doesn't know
        tfjs_config.pop('optional', None)

    return {
        'class_name':    class_name,
        'name':          name,
        'config':        tfjs_config,
        'inbound_nodes': convert_inbound_nodes(inbound_nodes),
    }


def build_tfjs_model_config(keras_config: dict) -> dict:
    """Build the modelTopology.config section for TF.js."""
    layers_raw = keras_config.get('layers', [])
    tfjs_layers = [build_tfjs_layer(l) for l in layers_raw]

    return {
        'name':          keras_config.get('name', 'model'),
        'trainable':     keras_config.get('trainable', True),
        'layers':        tfjs_layers,
        'input_layers':  keras_config.get('input_layers', []),
        'output_layers': keras_config.get('output_layers', []),
    }


# ── Main conversion ──────────────────────────────────────────────────────────

def convert(h5_path: str, out_dir: str):
    try:
        import tensorflow as tf
    except ImportError:
        logger.error("TensorFlow required"); sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Loading model: {h5_path}")
    model = tf.keras.models.load_model(h5_path)
    logger.info(f"Model loaded — {model.count_params():,} parameters")

    # ── 1. Collect & write weights ─────────────────────────────────────────
    weight_manifest_entries = []
    raw_buffers             = []

    for layer in model.layers:
        for w in layer.weights:
            arr         = w.numpy().astype(np.float32)
            base_name   = w.name.split(':')[0]
            name        = base_name if base_name.startswith(layer.name + '/') \
                          else f"{layer.name}/{base_name}"
            weight_manifest_entries.append({
                'name':  name,
                'shape': list(arr.shape),
                'dtype': 'float32',
            })
            raw_buffers.append(arr.flatten().tobytes())

    bin_name = 'weights.bin'
    bin_path = os.path.join(out_dir, bin_name)
    total_bytes = 0
    with open(bin_path, 'wb') as f:
        for buf in raw_buffers:
            f.write(buf)
            total_bytes += len(buf)
    logger.info(f"Weights binary: {total_bytes / 1024:.1f} KB → {bin_path}")

    # ── 2. Build topology ──────────────────────────────────────────────────
    keras_model_json = json.loads(model.to_json())
    keras_config     = keras_model_json.get('config', {})

    tfjs_model_config = build_tfjs_model_config(keras_config)

    model_json = {
        'format':      'layers-model',
        'generatedBy': 'custom-exporter-v3',
        'convertedBy': None,
        'modelTopology': {
            'class_name':    keras_model_json.get('class_name', 'Functional'),
            'config':        tfjs_model_config,
            'keras_version': '2.15.0',   # TF.js is happier with older version strings
            'backend':       'tensorflow',
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
        f.write(json.dumps(model_json, separators=(',', ':')))

    logger.info(f"Model JSON written → {json_path}")

    json_kb = os.path.getsize(json_path) / 1024
    bin_kb  = os.path.getsize(bin_path)  / 1024
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"  model.json : {json_kb:.1f} KB")
    logger.info(f"  weights.bin: {bin_kb:.1f} KB")
    logger.info("\n✅ TF.js model ready!")
    logger.info("\nStart the Flask app and visit http://localhost:5000/acoustic")


if __name__ == '__main__':
    if not os.path.exists(H5_PATH):
        logger.error(f"No trained model at {H5_PATH}. Run acoustic/train.py first.")
        sys.exit(1)
    convert(H5_PATH, TFJS_OUT)
