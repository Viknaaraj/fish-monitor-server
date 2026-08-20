from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process_data():
    data = request.json
    print("Received:", data)
    return jsonify({"status": "received", "you_sent": data})

@app.route("/")
def home():
    return "Server is running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)