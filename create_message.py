import os
import json
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def generate_messages(input_data: Dict, client):
    segments = input_data.get("segments", [])
    core_benefit_text = input_data.get("coreBenefitText", "")
    campaign_title = input_data.get("title", "프로모션")
    source_Url = input_data.get("sourceUrl", "")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 SK텔레콤, KT, 카드사 등에서
장기 이용 고객에게 발송되는
CRM 안내 문자를 작성하는 실무 카피라이터다.

- 메시지에는 반드시 "세그먼트 반영 문장" 1문장이 포함되어야 한다
- 이 문장은 고객이 왜 이 프로모션 대상이 되었는지를 직접적으로 드러내야 한다
- 세그먼트 반영 문장은 다음 유형 중 하나를 반드시 사용한다:
  • 연령 기반: "20대 고객님께 특히 필요한"
  • 이용 행태 기반: "평소 ○○을 자주 이용하시는 고객님께"
  • 관계 기반: "오랜 기간 함께해 주신 고객님께"
  • 라이프스타일 기반: "일상 혜택을 중요하게 생각하시는 고객님께"
- 추상적인 표현(‘특별한 고객님’, ‘소중한 고객님’)만 사용하는 것은 금지

당신의 메시지는 광고이지만
다음 원칙을 반드시 따른다:

- 혜택을 '안내'하는 톤 유지
- 고객과의 관계를 존중하는 문장 사용
- 실제 통신사/금융사에서 발송 가능한 문장만 사용

메시지는 정보 전달 중심이며,
차분하고 신뢰감 있는 톤으로 작성한다.

출력은 반드시 JSON만 허용한다.
"""),
        ("human", """
아래 정보를 바탕으로
통신사/금융사 CRM 안내 문자 1건을 작성하세요.
         
메시지 구성:
1. (광고) 헤더
2. 개인화 인사 + 안내 사유
3. 혜택 요약 문단
4. ■ 혜택 상세 안내 (조건 / 기간)
5. ▶ CTA 링크
6. 추가 안내 문장
7. ■ 문의처
8. 무료 수신거부 문구
     
중요 규칙 (반드시 지킬 것):
- {title} 값은 임의로 수정하거나 요약하지 말고,
  입력된 문자열을 그대로 메시지에 포함할 것
- {sourceUrl} 값은 반드시 메시지에 실제 URL 문자열 그대로 출력할 것
- "[링크]", "[URL]" 같은 대체 표현 사용 금지
- 입력값이 비어 있지 않은 경우, 누락은 오류로 간주

작성 규칙:
- 첫 줄은 반드시 다음 형식과 정확히 일치해야 함:
  (광고) {title} 안내
- 혜택은 객관적 사실 위주로 서술
- 문장은 짧고 단정하게 작성
CTA 규칙:
- 아래 문장을 그대로 사용할 것 (문구 수정 금지)
▶ 자세히 보기: {sourceUrl}
     
반드시 아래 문장을 메시지 초반(2~3번째 줄)에 포함할 것:
[세그먼트 반영 문장 작성 규칙]
- 타겟 세그먼트 설명을 바탕으로
- 고객이 "아, 이건 내 얘기네"라고 느낄 수 있게 작성
- 예시는 참고용이며 그대로 복사하지 말 것

예시:
- 20대 중심 세그먼트 → "20대 고객님의 라이프스타일에 맞춰 준비한 혜택입니다."
- 장기 이용 고객 → "오랜 기간 함께해 주신 고객님께 감사의 의미로 준비했습니다."
- 데이터 헤비 유저 → "데이터 사용이 잦은 고객님께 실질적인 혜택이 될 수 있도록 구성했습니다."
- 콘텐츠 소비형 → "영상·콘텐츠 이용이 잦은 고객님께 유용한 혜택을 안내드립니다."
         
혜택 문단 규칙:
- 반드시 여러 줄 목록 형태로 작성할 것
- 각 혜택은 "-" 로 시작하는 한 줄로 분리할 것
- 한 줄에 여러 혜택을 나열하는 문장은 금지

     
입력 정보:
프로모션명: {title}
혜택 설명: {coreBenefitText}
타겟 세그먼트 설명: {targetFeatures}
프로모션 url : {sourceUrl}

출력 형식:
{{
  "message_text": "..."
}}
""")
    ])

    chain = prompt | llm
    results = []

    for idx, segment in enumerate(segments):
        raw = chain.invoke({
            "title": campaign_title,
            "coreBenefitText": core_benefit_text,
            "sourceUrl": source_Url,
            "targetSegment": segment["target_segment"],
            "targetFeatures": segment["segment_features"]
        }).content

        content = raw.strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from LLM:\n{raw}") from e

        results.append({
            "target_group_index": idx,
            "target_name": segment["target_segment"],
            "message_drafts": [
                {
                    "message_draft_index": 1,
                    "message_text": parsed["message_text"]
                }
            ]
        })

    return {
        "title": campaign_title,
        "messages": results
    }
