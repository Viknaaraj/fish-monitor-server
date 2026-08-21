from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db
import base64
import numpy as np
import cv2

cred = credentials.Certificate("firebase-key.json")  # use your actual filename
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://fish-monitor-d1886-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process_data():
    data = request.json
    ref = db.reference("sensor_data")
    ref.push(data)
    return jsonify({"status": "saved to firebase"})

@app.route("/anomaly", methods=["POST"])
def anomaly_alert():
    data = request.json
    image_b64 = data.get("image")  # already base64 text from the Pi
    ref = db.reference("anomaly_alerts")
    ref.push({
        "timestamp": data.get("timestamp"),
        "reason": data.get("reason"),
        "image": image_b64
    })
    return jsonify({"status": "anomaly saved"})

@app.route("/")
def home():
    return "Server is running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
