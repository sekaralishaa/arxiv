from flask import Flask, request, jsonify
from recommend_utils import load_model, get_recommendation

app = Flask(__name__)
model = load_model()

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    query = data.get("query", "")
    print("📨 Received query:", query)  # Debug log

    if not query:
        print("⚠️ No input received")
        return jsonify({"error": "No input"}), 400

    top = get_recommendation(query, model)
    print("✅ Recommendation generated:", top[:1])  # Show only first result
    return jsonify(top)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
