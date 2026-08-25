import json
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import gc

cred_dict = json.loads(os.environ["FIREBASE_CREDENTIALS_JSON"])
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://fish-monitor-d1886-default-rtdb.asia-southeast1.firebasedatabase.app"
})
app = Flask(__name__)

with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# Global variables to store the camera matrix after the first calculation
newcameramtx = None
roi = None
rx = ry = rw = rh = 0

fgbg = cv2.createBackgroundSubtractorMOG2(history=15, varThreshold=50, detectShadows=False)
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

    # Calculate camera matrix ONCE on the first frame
    if newcameramtx is None:
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        rx, ry, rw, rh = roi

    # Reuse the matrix for undistortion
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    del img 
    
    dst = dst[ry:ry+rh, rx:rx+rw]

    scale_percent = 800.0 / dst.shape[1]
    new_width = 800
    new_height = int(dst.shape[0] * scale_percent)
    dst = cv2.resize(dst, (new_width, new_height))

    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    fgmask = fgbg.apply(blurred)
    
    cleaned_mask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

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

    os.remove(save_path)
    gc.collect()

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
