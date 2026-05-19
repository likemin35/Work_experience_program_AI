import json
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import OpenAI
from pydantic import BaseModel

from create_message import generate_messages
from strategy_agent import CampaignStrategyAgent
from upload_pdf import extract_text_from_pdf, parse_promotion_fields

app = FastAPI(title="Message AI Server")

pdf_client = OpenAI()
message_client = OpenAI()
cluster_client = OpenAI()
strategy_agent = CampaignStrategyAgent(client=OpenAI())


class CampaignPayload(BaseModel):
    title: Optional[str] = None
    purpose: Optional[str] = None
    coreBenefitText: Optional[str] = None


class CustomerPayload(BaseModel):
    customerId: str
    description: str


class ClusterCustomersRequest(BaseModel):
    campaign: CampaignPayload
    customers: List[CustomerPayload]


def _clean_json_block(text: str) -> str:
    content = (text or "").strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()
    return content


@app.post("/generate-messages")
def generate_messages_api(data: Dict[str, Any]):
    if not data:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON input"},
        )

    try:
        result = generate_messages(
            data,
            client=message_client,
            strategy_agent=strategy_agent,
        )
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/cluster-customers")
def cluster_customers(data: ClusterCustomersRequest):
    campaign = data.campaign.model_dump(exclude_none=True)
    customers = [customer.model_dump() for customer in data.customers]

    if not campaign or not customers:
        return JSONResponse(
            status_code=400,
            content={"error": "campaign or customers missing"},
        )

    clustering_strategy = strategy_agent.decide_clustering_strategy(
        campaign=campaign,
        customers=customers,
    )

    customer_lines = "\n".join(
        f"- ({customer.get('customerId')}) {customer.get('description', '')}"
        for customer in customers
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

        parsed = json.loads(
            _clean_json_block(response.choices[0].message.content)
        )
        clusters = parsed.get("clusters")
        if not isinstance(clusters, list):
            return JSONResponse(
                status_code=500,
                content={"error": "Invalid cluster response format"},
            )

        for cluster in clusters:
            cluster["customerIds"] = [
                str(cid) for cid in cluster.get("customerIds", [])
            ]

        return JSONResponse(
            status_code=200,
            content={
                "clusters": clusters,
                "strategy_meta": clustering_strategy,
            },
        )
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "cluster parsing failed"},
        )


@app.post("/ai/campaign/extract")
async def extract_campaign(file: Optional[UploadFile] = File(default=None)):
    try:
        if file is None:
            return JSONResponse(
                status_code=400,
                content={"error": "file missing"},
            )

        file_bytes = await file.read()
        pdf_text = extract_text_from_pdf(file_bytes)
        result = parse_promotion_fields(pdf_text, client=pdf_client)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
def health_check():
    return PlainTextResponse("AI Cluster Server running", status_code=200)
