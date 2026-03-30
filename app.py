import json
import traceback

from flask import Flask, jsonify, request
from openai import OpenAI

from create_message import generate_messages
from strategy_agent import CampaignStrategyAgent
from upload_pdf import extract_text_from_pdf, parse_promotion_fields

app = Flask(__name__)

pdf_client = OpenAI()
message_client = OpenAI()
cluster_client = OpenAI()
strategy_agent = CampaignStrategyAgent(client=OpenAI())


def _clean_json_block(text: str):
    content = (text or "").strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()
    return content


@app.route("/generate-messages", methods=["POST"])
def generate_messages_api():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON input"}), 400

    try:
        result = generate_messages(
            data,
            client=message_client,
            strategy_agent=strategy_agent,
        )
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

    clustering_strategy = strategy_agent.decide_clustering_strategy(
        campaign=campaign,
        customers=customers,
    )

    customer_lines = "\n".join(
        f"- ({c.get('customerId')}) {c.get('description', '')}"
        for c in customers
    )

    prompt = f"""
You are a telecom CRM segmentation strategist.
Group customers for the campaign.

[Campaign]
title: {campaign.get("title")}
benefit: {campaign.get("coreBenefitText")}

[Strategy from agent]
mode: {clustering_strategy.get("strategy_mode")}
reason: {clustering_strategy.get("reason")}
guidance: {clustering_strategy.get("segmentation_guidance")}
reference promotion ids: {clustering_strategy.get("history_references")}

Rules:
- Choose a reasonable number of groups.
- If users are all very different, single-user groups are allowed.
- Return JSON only in this shape:
{{
  "clusters": [
    {{
      "clusterName": "string",
      "clusterDescription": "string",
      "customerIds": ["1", "2"]
    }}
  ]
}}

[Customers]
{customer_lines}
"""

    try:
        response = cluster_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        parsed = json.loads(_clean_json_block(response.choices[0].message.content))
        clusters = parsed.get("clusters")
        if not isinstance(clusters, list):
            return jsonify({"error": "Invalid cluster response format"}), 500

        for cluster in clusters:
            cluster["customerIds"] = [str(cid) for cid in cluster.get("customerIds", [])]

        return jsonify({
            "clusters": clusters,
            "strategy_meta": clustering_strategy,
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
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/")
def health_check():
    return "AI Cluster Server running", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
