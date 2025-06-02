from flask import Flask, request, jsonify
from recommend_utils import load_model_and_data, get_recommendation

app = Flask(__name__)
df, vec_all, model = load_model_and_data()

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No input"}), 400

    top = get_recommendation(query, df, vec_all, model)
    return jsonify(top)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
