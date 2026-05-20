"""
test_api.py - Quick end-to-end test of the /predict endpoint.
Run while Flask server is active:  python test_api.py
"""
import requests, base64, json, sys

IMG_PATH = r"test_images\test_0_actinic_keratoses.jpg"
URL      = "http://127.0.0.1:5000/predict"

with open(IMG_PATH, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {"image_b64": f"data:image/jpeg;base64,{b64}"}
print(f"Sending request to {URL} ...")
r = requests.post(URL, json=payload, timeout=120)
data = r.json()

if r.status_code != 200 or data.get("status") != "success":
    print("ERROR:", json.dumps(data, indent=2))
    sys.exit(1)

print("\n=== PREDICTION RESULT ===")
print(f"Status      : {data.get('status')}")
print(f"Prediction  : {data.get('display_name')}")
print(f"Category    : {data.get('category')}")
print(f"Confidence  : {data.get('confidence')}%")
print(f"Risk Level  : {data.get('risk_level')}")
print(f"Model Used  : {data.get('model_used')}")
print(f"Image Type  : {data.get('image_type')}")
print(f"Heatmap     : {'YES (' + str(len(data.get('heatmap', ''))) + ' chars)' if data.get('heatmap') else 'MISSING'}")
print(f"Orig Image  : {'YES' if data.get('original_image') else 'MISSING'}")
print(f"Rec Summary : {data.get('recommendation', {}).get('summary', 'N/A')}")

print("\nAll Class Probabilities:")
probs = data.get("all_probs", {})
for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
    bar = "#" * int(prob / 3)
    print(f"  {cls:<40} {prob:5.1f}%  {bar}")

print("\nAll checks PASSED." if all([
    data.get("heatmap"),
    data.get("original_image"),
    data.get("recommendation"),
    data.get("all_probs"),
]) else "\nWARNING: Some fields missing!")
