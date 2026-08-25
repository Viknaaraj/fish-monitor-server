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

# Tuned MOG2: higher varThreshold (40) ignores minor bubble brightness flickers
fgbg = cv2.createBackgroundSubtractorMOG2(history=15, varThreshold=40, detectShadows=False)

# Morphological kernel for removing bubble/food noise
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

    # 1. Undistort and crop to region of interest
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    rx, ry, rw, rh = roi
    dst = dst[ry:ry+rh, rx:rx+rw]

    # 2. Pre-filter noise with Gaussian blur
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. Apply background subtraction
    fgmask = fgbg.apply(blurred)

    # 4. Remove small bubbles/food particles using morphological opening and closing
    cleaned_mask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Extract contours and filter by strict fish dimensions
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_fish = []
    for c in contours:
        area = cv2.contourArea(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        
        # Filter: Ignore small speckles (< 1200px) or unrealistically thin contours
        if 1200 < area < 50000 and bw >= 25 and bh >= 25:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                valid_fish.append({"x": cx, "y": cy, "area": int(area)})

    os.remove(save_path)

    if valid_fish:
        ref = db.reference("fish_positions")
        ref.push({
            "timestamp": datetime.utcnow().isoformat(),
            "frame": file.filename,
            "detected_count": len(valid_fish),
            "positions": valid_fish
        })
        return jsonify({"status": "tracked", "fish_count": len(valid_fish), "positions": valid_fish})
    else:
        return jsonify({"status": "no fish detected in frame"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
