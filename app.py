import json
import os
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db

cred_dict = json.loads(os.environ["FIREBASE_CREDENTIALS_JSON"])
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://fish-monitor-d1886-default-rtdb.asia-southeast1.firebasedatabase.app"
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

    print(f"Received and saved: {file.filename}")
    return jsonify({"status": "received", "filename": file.filename})

# This must always be the very last section of the file
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
