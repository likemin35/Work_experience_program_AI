import io
import json
import pdfplumber
from openai import OpenAI

PROMPT_TEMPLATE = """
너는 통신사 프로모션 문서를 구조화해서 읽는 시스템이다.
아래 PDF 문서 전체를 읽고, 정의에 맞는 정보만 추출하라.

반드시 JSON만 출력하라.
설명, 문장, 코드블록, 주석 금지.

[필드 정의]
- title: 프로모션의 공식 명칭 또는 문서에서 가장 대표적인 제목
- coreBenefitText: 프로모션 내용 및 고객에게 제공되는 주요 혜택(조건, 할인, 제공 내용 포함)
- sourceUrl: 프로모션 참여 또는 상세 안내를 위한 공식 URL (문서에 없으면 null)

[출력 형식]
{{
  "title": string,
  "coreBenefitText": string,
  "sourceUrl": string | null
}}

[문서 내용]
{text}
"""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    texts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return "\n".join(texts)


def parse_promotion_fields(pdf_text: str, client: OpenAI) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "너는 문서 정보 추출기다."},
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(text=pdf_text)
            }
        ]
    )

    return json.loads(response.choices[0].message.content)
