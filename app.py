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

with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

fgbg = cv2.createBackgroundSubtractorMOG2(history=15, varThreshold=25, detectShadows=False)

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
        return jsonify({"status": "error reading image"}), 400

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w_roi, h_roi = roi
    dst = dst[y:y+h_roi, x:x+w_roi]

    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fish_sized = [c for c in contours if cv2.contourArea(c) > 500]
    position = None
    if fish_sized:
        largest = max(fish_sized, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            position = {"x": cx, "y": cy}

    os.remove(save_path)  # done with the image, don't keep filling /tmp

    if position:
        ref = db.reference("fish_positions")
        ref.push({
            "timestamp": datetime.utcnow().isoformat(),
            "frame": file.filename,
            "x": position["x"],
            "y": position["y"]
        })
        return jsonify({"status": "tracked", "position": position})
    else:
        return jsonify({"status": "no fish detected in frame"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
