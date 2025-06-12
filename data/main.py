
from flask import Flask, request, jsonify
from recommend_utils import load_model, recommend_articles

app = Flask(__name__)
model = load_model()

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()

    user_title = data.get("title", "").strip()
    user_keywords = data.get("keywords", "").strip()
    user_category = data.get("category", "").strip()

    print("📨 Received input:")
    print("📝 Title:", user_title)
    print("🔑 Keywords:", user_keywords)
    print("📂 Category:", user_category)

    if not any([user_title, user_keywords, user_category]):
        return jsonify({"error": "❌ Minimal salah satu input harus diisi."}), 400

    try:
        top = recommend_articles(user_title, user_keywords, user_category, model)
        print("✅ Recommendation generated")
        return jsonify(top)
    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
