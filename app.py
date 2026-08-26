import json
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime, timezone, timedelta
import gc
import joblib

# Initialize Firebase Admin SDK
cred_dict = json.loads(os.environ["FIREBASE_CREDENTIALS_JSON"])
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://fish-monitor-d1886-default-rtdb.asia-southeast1.firebasedatabase.app"
})

app = Flask(__name__)

with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

water_model = joblib.load("water_quality_model.pkl")

try:
    behavior_model = joblib.load("behavior_model.pkl")
except Exception as e:
    behavior_model = None
    print(f"Behavior model not loaded: {e}")

newcameramtx = None
roi = None
rx = ry = rw = rh = 0

# Lowered threshold to 16 to detect faint fish colors, increased history
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)
open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

@app.route("/")
def home():
    return "Server is running"

@app.route("/classify_water", methods=["POST"])
def classify_water():
    data = request.json
    temp = data.get("temp")
    turbidity_band_text = data.get("turbidity_band")
    ph = data.get("ph")

    band_map = {"Clear": 0, "Slightly turbid": 1, "Turbid": 2, "Very turbid": 3}
    turbidity_band = band_map.get(turbidity_band_text, 1)

    prediction = water_model.predict([[temp, turbidity_band, ph]])[0]
    labels = {0: "Excellent", 1: "Good", 2: "Poor"}
    result = labels.get(prediction, "Unknown")

    ref = db.reference("water_quality_status")
    ref.push({
        "timestamp": data.get("timestamp"),
        "temp": temp,
        "ph": ph,
        "turbidity_band": turbidity_band_text,
        "quality": result
    })

    return jsonify({"water_quality": result, "status": "saved to firebase"})

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

    if newcameramtx is None:
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        rx, ry, rw, rh = roi

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    del img
    dst = dst[ry:ry+rh, rx:rx+rw]

    scale_percent = 800.0 / dst.shape[1]
    new_width = 800
    new_height = int(dst.shape[0] * scale_percent)
    dst = cv2.resize(dst, (new_width, new_height))

    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    mask_top = int(new_height * 0.25)
    blurred[0:mask_top, :] = 0
    mask_left = int(new_width * 0.25)
    blurred[:, 0:mask_left] = 0
    
    # Lowered learning rate so slow fish do not disappear
    fgmask = fgbg.apply(blurred, learningRate=0.005)

    cleaned_mask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    cv2.imwrite("/tmp/latest_mask.jpg", cleaned_mask)

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

    my_tz = timezone(timedelta(hours=8))
    local_time = datetime.now(my_tz).isoformat()
    is_anomaly = False

    if position and behavior_model is not None:
        prediction = behavior_model.predict([[position["x"], position["y"], position["area"]]])[0]
        if prediction == -1:
            is_anomaly = True
            anomaly_ref = db.reference("anomaly_alerts")
            anomaly_ref.push({
                "timestamp": local_time,
                "frame": file.filename,
                "reason": "Abnormal behavior detected (location or size outlier)",
                "x": position["x"],
                "y": position["y"]
            })

    if os.path.exists(save_path):
        os.remove(save_path)
    gc.collect()

    if position:
        ref = db.reference("fish_positions")
        ref.push({
            "timestamp": local_time,
            "frame": file.filename,
            "x": position["x"],
            "y": position["y"],
            "area": position["area"],
            "is_anomaly": is_anomaly
        })
        return jsonify({"status": "tracked", "position": position, "anomaly": is_anomaly})
    else:
        return jsonify({"status": "no fish detected in frame"})

@app.route("/debug/mask")
def view_mask():
    if os.path.exists("/tmp/latest_mask.jpg"):
        return send_file("/tmp/latest_mask.jpg", mimetype="image/jpeg")
    else:
        return "No mask generated yet. Run the Pi script first."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
