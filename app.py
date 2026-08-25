import json
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import gc

# Initialize Firebase Admin SDK
cred_dict = json.loads(os.environ["FIREBASE_CREDENTIALS_JSON"])
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://fish-monitor-d1886-default-rtdb.asia-southeast1.firebasedatabase.app"
})

app = Flask(__name__)

# Load calibration matrices once on server startup
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# Cached camera calibration parameters to avoid recalculating per request
newcameramtx = None
roi = None
rx = ry = rw = rh = 0

# Tuned MOG2 background subtractor:
# - history=60: Remembers background across longer cycles
# - varThreshold=25: Sensitive enough to capture subtle movements
fgbg = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=25, detectShadows=False)

# Morphological kernels
open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))   # Removes micro-bubbles & food speckles
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Bridges gaps in the fish contour

@app.route("/process", methods=["POST"])
def process_data():
    """Receives and stores analog sensor readings in Firebase."""
    data = request.json
    ref = db.reference("sensor_data")
    ref.push(data)
    return jsonify({"status": "saved to firebase"})

@app.route("/anomaly", methods=["POST"])
def anomaly_alert():
    """Receives and logs anomaly flags and alert frames."""
    data = request.json
    ref = db.reference("anomaly_alerts")
    ref.push({
        "timestamp": data.get("timestamp"),
        "reason": data.get("reason"),
        "image": data.get("image")
    })
    return jsonify({"status": "anomaly saved"})

@app.route("/")
def home():
    return "Server is running"

@app.route("/upload", methods=["POST"])
def upload_image():
    global newcameramtx, roi, rx, ry, rw, rh

    if 'file' not in request.files:
        return jsonify({"error": "no file provided"}), 400

    file = request.files['file']
    save_path = f"/tmp/{file.filename}"
    file.save(save_path)

    img = cv2.imread(save_path)
    if img is None:
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({"status": "error reading image"}), 400

    # 1. Compute optimal camera matrix once on the first frame received
    if newcameramtx is None:
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        rx, ry, rw, rh = roi

    # 2. Undistort and crop at full native resolution
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    del img  # Free raw full-size image from memory immediately
    dst = dst[ry:ry+rh, rx:rx+rw]

    # 3. Downscale after undistortion to stay within memory limits
    scale_percent = 800.0 / dst.shape[1]
    new_width = 800
    new_height = int(dst.shape[0] * scale_percent)
    dst = cv2.resize(dst, (new_width, new_height))

    # 4. Noise filtering & background subtraction
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # learningRate=0.01 prevents slow-moving or resting fish from blending into background
    fgmask = fgbg.apply(blurred, learningRate=0.01)

    # 5. Clean up mask: eliminate bubbles without eroding the fish body
    cleaned_mask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # 6. Extract contours and track the single largest moving target
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fish_contours = [c for c in contours if cv2.contourArea(c) > 150]
    position = None

    if fish_contours:
        largest = max(fish_contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            position = {"x": cx, "y": cy, "area": int(cv2.contourArea(largest))}

    # Clean up disk and RAM
    if os.path.exists(save_path):
        os.remove(save_path)
    gc.collect()

    # 7. Push coordinates to Firebase
    if position:
        ref = db.reference("fish_positions")
        ref.push({
            "timestamp": datetime.utcnow().isoformat(),
            "frame": file.filename,
            "x": position["x"],
            "y": position["y"],
            "area": position["area"]
        })
        return jsonify({"status": "tracked", "position": position})
    else:
        return jsonify({"status": "no fish detected in frame"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
