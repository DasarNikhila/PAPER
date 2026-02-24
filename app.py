from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import pickle
import json

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import threading
import time
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ===============================
# CONFIG
# ===============================
IMAGE_SIZE = (224, 224)
CLASSES = ["Black Rot", "Healthy", "Insect Hole"]
CONFIDENCE_THRESHOLD = 50

TRAIN_DIR = os.path.join("dataset", "train")
TEST_DIR  = os.path.join("dataset", "test")

EVAL_FILE = "evaluation_results.json"

# evaluation background state
eval_in_progress = False
eval_lock = threading.Lock()

# ===============================
# DISEASE DESCRIPTIONS
# ===============================
DISEASE_DESCRIPTIONS = {
    "Healthy": (
        "A healthy cauliflower leaf is uniformly green with a smooth texture. "
        "There are no holes, lesions, or yellowing."
    ),
    "Black Rot": (
        "Black rot is a bacterial disease causing V-shaped yellow lesions "
        "and darkened veins under humid conditions."
    ),
    "Insect Hole": (
        "Insect hole damage appears as irregular holes caused by pest feeding."
    )
}

# ===============================
# DISEASE TREATMENTS / RECOMMENDATIONS
# ===============================
DISEASE_TREATMENTS = {
    "Healthy": """
No treatment required. Maintain good cultural practices: proper watering, fertilization, and crop rotation to prevent disease.
""",

    "Black Rot": """
Remove and destroy infected leaves; improve air circulation; avoid overhead irrigation. Use certified disease-free transplants and rotate crops. If bacterial spread is severe, consider copper-based bactericides following local regulations.
""",

    "Insect Hole": """
Inspect for pests and remove by hand when feasible. Use insecticidal soaps or neem oil for control. Encourage beneficial insects and maintain field sanitation. Use targeted pesticides only if thresholds are exceeded.
"""
}

# ===============================
# LOAD FEATURE EXTRACTOR
# ===============================
feature_extractor = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

# ===============================
# LOAD SVM + SCALER
# ===============================
if not os.path.exists("svm_model.pkl") or not os.path.exists("scaler.pkl"):
    raise FileNotFoundError("Run train_svm.py first.")

svm = pickle.load(open("svm_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_features(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    feat = feature_extractor.predict(img, verbose=0)
    feat = scaler.transform(feat)

    return feat[0]

# ===============================
# MAIN ROUTE
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    # Collect results for one or more uploaded images
    results = []
    # keep legacy single-value variables for template compatibility
    prediction = confidence = description = image_path = None
    top_predictions = []
    warning = None
    eval_results = None
    treatment = None

    if request.method == "POST":
        files = request.files.getlist("leaf")

        for file in files:
            if not file or file.filename == "":
                continue

            os.makedirs("static", exist_ok=True)
            filename = secure_filename(file.filename)
            unique_name = f"{int(time.time())}_{uuid.uuid4().hex}_{filename}"
            image_path = os.path.join("static", unique_name)
            file.save(image_path)

            feat = extract_features(image_path)
            probs = svm.predict_proba([feat])[0]

            best_idx = np.argmax(probs)
            best_conf = probs[best_idx] * 100

            if best_conf < CONFIDENCE_THRESHOLD:
                pred = CLASSES[best_idx]
                conf = round(best_conf, 2)
                desc = "Low confidence. Upload clearer image."
                warn = "Low confidence"
                treat = DISEASE_TREATMENTS.get(pred)
            else:
                pred = CLASSES[best_idx]
                conf = round(best_conf, 2)
                desc = DISEASE_DESCRIPTIONS[pred]
                warn = None
                treat = DISEASE_TREATMENTS.get(pred)

            top_preds = []
            for i, cls in enumerate(CLASSES):
                top_preds.append({
                    "class": cls,
                    "prob": round(float(probs[i]) * 100, 2)
                })

            results.append({
                "image_path": image_path,
                "prediction": pred,
                "confidence": conf,
                "description": desc,
                "treatment": treat,
                "top_predictions": top_preds,
                "warning": warn
            })

    # Ensure evaluation is being computed in background if missing
    try:
        if not os.path.exists(EVAL_FILE):
            # start background computation (non-blocking)
            start_evaluation_background()
    except Exception as e:
        print("⚠️ Evaluation background start failed:", e)

    if os.path.exists(EVAL_FILE):
        try:
            with open(EVAL_FILE, "r") as f:
                eval_results = json.load(f)
        except Exception as e:
            print("⚠️ Failed to read evaluation file:", e)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        description=description,
        image_path=image_path,
        top_predictions=top_predictions,
        warning=warning,
        treatment=treatment,
        eval_results=eval_results,
        results=results
    )

# ===============================
# COMPUTE & SAVE EVALUATION ONCE
# ===============================
def compute_and_save_evaluation():
    global eval_in_progress
    with eval_lock:
        if eval_in_progress:
            return
        eval_in_progress = True

    print("🔹 Computing evaluation (background)...")

    X_test = []
    y_test = []

    for label, cls in enumerate(CLASSES):
        folder = os.path.join(TEST_DIR, cls)

        images = [f for f in os.listdir(folder)
                  if f.lower().endswith(("jpg", "jpeg", "png"))]

        print(cls, "->", len(images), "images")

        for img_name in images:
            img_path = os.path.join(folder, img_name)
            feat = extract_features(img_path)
            X_test.append(feat)
            y_test.append(label)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    y_pred = svm.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    p, r, f1, support = precision_recall_fscore_support(
        y_test, y_pred, zero_division=0
    )

    metrics = []
    for i, cls in enumerate(CLASSES):
        metrics.append({
            "class": cls,
            "precision": round(float(p[i]), 3),
            "recall": round(float(r[i]), 3),
            "f1": round(float(f1[i]), 3),
            "support": int(support[i])
        })

    results = {
        "accuracy": round(float(acc), 4),
        "metrics": metrics,
        "confusion_matrix": cm.tolist()
    }

    with open(EVAL_FILE, "w") as f:
        json.dump(results, f)
    print("✅ Evaluation saved to file")

    # mark finished
    with eval_lock:
        eval_in_progress = False


def start_evaluation_background():
    """Start compute_and_save_evaluation in a background thread if not running."""
    global eval_in_progress
    with eval_lock:
        if eval_in_progress or os.path.exists(EVAL_FILE):
            return
        eval_in_progress = True

    def _runner():
        try:
            compute_and_save_evaluation()
        finally:
            with eval_lock:
                # ensure flag cleared even on error
                eval_in_progress = False

    t = threading.Thread(target=_runner, daemon=True)
    t.start()

# ===============================
# FAST EVALUATION ROUTE
# ===============================
@app.route("/evaluation")
def evaluation():

    # If evaluation file exists, return it immediately
    if os.path.exists(EVAL_FILE):
        with open(EVAL_FILE, "r") as f:
            results = json.load(f)
        return jsonify(results)

    # If computation already running, return status
    with eval_lock:
        if eval_in_progress:
            return jsonify({"status": "computing"})

    # Not running and no file -> start background job and return started
    start_evaluation_background()
    return jsonify({"status": "started"})

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
