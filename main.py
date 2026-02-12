from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import json

app = FastAPI()

cluster_client = OpenAI()

# ---------- Request DTO ----------

class Customer(BaseModel):
    customerId: str
    description: str

class Campaign(BaseModel):
    purpose: str
    coreBenefitText: str

class ClusterRequest(BaseModel):
    campaign: Campaign
    customers: List[Customer]

# ---------- Response DTO ----------

class ClusterResult(BaseModel):
    clusterName: str
    clusterDescription: str
    customerIds: List[str]

class ClusterResponse(BaseModel):
    clusters: List[ClusterResult]

# ---------- API ----------

@app.post("/cluster-customers", response_model=ClusterResponse)
def cluster_customers(req: ClusterRequest):

    prompt = f"""
너는 통신사 마케팅 전문가다.

[캠페인 정보]
- 목적: {req.campaign.purpose}
- 핵심 혜택: {req.campaign.coreBenefitText}

아래 고객들을
이 캠페인 기준으로 의미 있는 타겟 그룹으로 분류하라.

규칙:
- 그룹 개수는 네가 판단
- 모두 다르면 1명 = 1그룹 허용
- 반드시 아래 JSON 스키마로만 출력

출력 형식:
{{
  "clusters": [
    {{
      "clusterName": "그룹 이름",
      "clusterDescription": "이 그룹의 특징",
      "customerIds": ["C001", "C002"]
    }}
  ]
}}

[고객 목록]
""" + "\n".join(
        f"- ({c.customerId}) {c.description}"
        for c in req.customers
    )

    response = cluster_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(content)
    except Exception:
        raise ValueError("AI 응답 JSON 파싱 실패")

    return parsed
