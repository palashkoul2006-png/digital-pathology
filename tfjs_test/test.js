const tf = require('@tensorflow/tfjs');
async function run() {
  try {
    const modelUrl = 'file://../static/acoustic_model/model.json';
    const model = await tf.loadLayersModel(modelUrl);
    console.log("Model loaded successfully!");
  } catch(err) {
    console.error("FAILED:");
    console.error(err);
  }
}
run();
