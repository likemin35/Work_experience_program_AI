import json
from flask import Flask, request, jsonify
from openai import OpenAI
from create_message import generate_messages
from upload_pdf import extract_text_from_pdf, parse_promotion_fields
import traceback

app = Flask(__name__)

pdf_client = OpenAI()
message_client = OpenAI()
cluster_client = OpenAI()

@app.route("/generate-messages", methods=["POST"])
def generate_messages_api():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON input"}), 400

    try:
        result = generate_messages(data, client=message_client)
        return jsonify(result), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/cluster-customers", methods=["POST"])
def cluster_customers():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON input"}), 400

    campaign = data.get("campaign")
    customers = data.get("customers")

    if not campaign or not customers:
        return jsonify({"error": "campaign or customers missing"}), 400

    prompt = f"""
너는 통신사 마케팅 전문가다.

[캠페인 정보]
- 목적: {campaign.get('purpose')}
- 핵심 혜택: {campaign.get('coreBenefitText')}

아래 고객들을 이 캠페인 기준으로
의미 있는 타겟 그룹으로 분류하라.

규칙:
- 그룹 개수는 네가 판단
- 고객이 모두 다르면 1명 = 1그룹 허용
- 각 그룹마다
  - clusterName
  - clusterDescription
  - customerIds 포함

[고객 목록]
""" + "\n".join(
        f"- ({c['customerId']}) {c['description']}"
        for c in customers
    ) + """

JSON 형식으로만 출력하라.
"""

    try:
        response = cluster_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(content)

        # ✅ 핵심 수정: list / dict 모두 처리
        if isinstance(parsed, list):
            clusters = parsed
        elif isinstance(parsed, dict):
            if "clusters" in parsed:
                clusters = parsed["clusters"]
            elif "targetGroups" in parsed:
                clusters = parsed["targetGroups"]
            else:
                return jsonify({"error": "Invalid cluster response format"}), 500
        else:
            return jsonify({"error": "Invalid cluster response type"}), 500

        # customerIds 문자열 보정
        for cluster in clusters:
            cluster["customerIds"] = [
                str(cid) for cid in cluster.get("customerIds", [])
            ]

        # ✅ Spring이 기대하는 형태로 통일
        return jsonify({
            "clusters": clusters
        }), 200

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "cluster parsing failed"}), 500




@app.route("/ai/campaign/extract", methods=["POST"])
def extract_campaign():
    try:
        if "file" not in request.files:
            return jsonify({"error": "file missing"}), 400

        file = request.files["file"]
        file_bytes = file.read()

        pdf_text = extract_text_from_pdf(file_bytes)

        result = parse_promotion_fields(pdf_text, client=pdf_client)

        return jsonify(result), 200

    except Exception:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/")
def health_check():
    return "AI Cluster Server running", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
