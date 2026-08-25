import json
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

cred_dict = json.loads(os.environ["FIREBASE_CREDENTIALS_JSON"])
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://fish-monitor-d1886-default-rtdb.asia-southeast1.firebasedatabase.app"
})
app = Flask(__name__)

# Load camera calibration matrices
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# Background subtractor tuned for single fish motion
fgbg = cv2.createBackgroundSubtractorMOG2(history=15, varThreshold=50, detectShadows=False)

# Morphological kernel to erase bubble noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

@app.route("/process", methods=["POST"])
def process_data():
    data = request.json
    ref = db.reference("sensor_data")
    ref.push(data)
    return jsonify({"status": "saved to firebase"})

@app.route("/anomaly", methods=["POST"])
def anomaly_alert():
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

    # 1. Undistort and crop
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    rx, ry, rw, rh = roi
    dst = dst[ry:ry+rh, rx:rx+rw]

    # 2. Gaussian blur to remove high-frequency bubble noise
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # 3. Background subtraction & morphological filtering
    fgmask = fgbg.apply(blurred)
    cleaned_mask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Find contours and select only the single largest fish
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fish_contours = [c for c in contours if cv2.contourArea(c) > 1500]
    position = None

    if fish_contours:
        # Pick the largest contour (the fish)
        largest = max(fish_contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            position = {"x": cx, "y": cy, "area": int(cv2.contourArea(largest))}

    os.remove(save_path)

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
