import json
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase
cred_dict = json.loads(os.environ["FIREBASE_CREDENTIALS_JSON"])
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://fish-monitor-d1886-default-rtdb.asia-southeast1.firebasedatabase.app"
})

app = Flask(__name__)

# Load camera calibration matrices into memory at startup
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

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

    # Read image, apply calibration, and crop
    img = cv2.imread(save_path)
    if img is not None:
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        x, y, w_roi, h_roi = roi
        dst = dst[y:y+h_roi, x:x+w_roi]
        
        # Overwrite the raw temporary file with the flattened version
        cv2.imwrite(save_path, dst)
        print(f"Undistorted and saved: {file.filename}")
    else:
        print(f"Error: Could not read {file.filename} for undistortion.")

    return jsonify({"status": "received and undistorted", "filename": file.filename})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
