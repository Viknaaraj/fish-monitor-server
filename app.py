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

# Initialize the global background subtractor for movement tracking
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

    # Read image, apply calibration, and crop
    img = cv2.imread(save_path)
    if img is not None:
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        x, y, w_roi, h_roi = roi
        dst = dst[y:y+h_roi, x:x+w_roi]
        
        # Detect movement
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                bx, by, bw, bh = cv2.boundingRect(contour)
                cv2.rectangle(dst, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                
        # Overwrite the raw temporary file with the tracked version
        cv2.imwrite(save_path, dst)
        print(f"Tracked and saved: {file.filename}")
    else:
        print(f"Error: Could not read {file.filename} for processing.")

    return jsonify({"status": "received, undistorted, and tracked", "filename": file.filename})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
