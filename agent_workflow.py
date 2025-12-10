import os
from typing import List, Dict, TypedDict, Union
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from rag_utils import query_chroma
from rag_utils_target import query_chroma_targeting

# Pydantic 모델 정의 (LLM의 구조화된 출력을 위해)
class Persona(BaseModel):
    target_group_index: int = Field(description="타겟 그룹의 순번")
    target_name: str = Field(description="타겟 세그먼트의 이름")
    target_features: str = Field(description="타겟 세그먼트의 주요 특징")
    classification_reason: str = Field(description="이 세그먼트를 분류한 데이터 기반의 근거")

class Personas(BaseModel):
    personas: List[Persona]

# 1. State 구현: CampaignState TypedDict
class CampaignState(TypedDict):
    """
    LangGraph의 상태를 정의하는 TypedDict.
    모든 Agent가 공유하는 중앙 데이터 구조입니다.
    """
    input_data: Dict # BE 서버로부터의 초기 요청 데이터 (예: core_benefit_text, custom_columns 등)
    target_personas: Union[List[Dict], None] # Targeting Agent의 타겟 5개 분류 결과
    messages_drafts: Union[List[Dict], None] # Messaging Agent의 타겟별 초안 2개 생성 결과
    validation_reports: Union[List[Dict], None] # Validator Agent의 초안 검증 리포트
    rework_count: int # 메시지 재생성 시도 횟수 (무한 루프 방지용)
    refine_feedback: Union[Dict, None] # 마케터의 재요청 피드백
    final_output: Union[Dict, None] # Formatter Agent의 최종 결과

# RAG Tool 구현
def rag_search(query: str, source_type: str) -> str:
    """
    RAG (Retrieval Augmented Generation) 툴입니다.
    query_chroma를 호출하여 Knowledge_Base DB 및 벡터 저장소에서 관련 지식을 검색하고,
    결과를 LLM 프롬프트에 포함하기 좋은 단일 문자열로 포맷팅합니다.

    Args:
        query (str): 검색할 쿼리.
        source_type (str): 검색할 지식의 출처 타입 (예: '정책', '성공 사례', '스팸/광고 정책').

    Returns:
        str: 검색된 관련 지식 요약 문자열.
    """
    print(f"RAG Search Called - Query: '{query}', Source Type: '{source_type}'")
    
    # rag_utils의 query_chroma 함수를 사용하여 ChromaDB에서 검색
    search_results = query_chroma(
        query_texts=[query],
        n_results=3, # 관련성 높은 3개 결과 사용
        where_filter={"source_type": source_type}
    )
    
    if not search_results:
        return "관련 지식을 찾을 수 없습니다."
    
    # 검색 결과를 단일 문자열로 포맷팅
    formatted_knowledge = "\n".join([
        f"- {result['document']} (출처: {result['metadata'].get('title', 'N/A')}, 관련성 점수: {1-result['distance']:.2f})"
        for result in search_results
    ])
    
    return f"'{source_type}' 관련 검색된 지식:\n{formatted_knowledge}"


def rag_search_targeting(query: str) -> str:
    """
    소비자 세그먼트 논문 DB용 RAG 검색.
    """
    print(f"Targeting RAG Search Called - Query: '{query}'")

    results = query_chroma_targeting(
        query_texts=[query],
        n_results=5,
        where_filter=None  # 논문 메타데이터 필터 필요 시 추가 가능
    )

    if not results:
        return "관련 세그먼트 지식을 찾을 수 없습니다."

    formatted = "\n".join([
        f"- {r['document']} (출처: {r['metadata'].get('title', 'N/A')}, 점수: {1-r['distance']:.2f})"
        for r in results
    ])

    return f"[세그먼트 관련 지식]\n{formatted}"


# 2. Agent 함수 구현 (LLM 연동)

# LLM, Parser, Prompt 등 공통 컴포넌트 초기화
# 참고: OpenAI API 키는 환경변수 'OPENAI_API_KEY'에 설정되어 있어야 합니다.
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))
json_parser = JsonOutputParser()

def run_targeting_agent(state: CampaignState) -> Dict:
    """
    Targeting Agent: 마케터의 핵심 혜택을 기반으로 5개의 상이한 타겟 페르소나를 분류합니다.
    상태에 이미 페르소나가 존재하면, 해당 페르소나를 그대로 사용합니다.
    """
    print("---" + " Targeting Agent 실행 중 ---")
    
    # 상태에 이미 페르소나가 존재하면, 해당 페르소나를 그대로 사용하고 다음 단계로 넘어갑니다.
    if state.get('target_personas'):
        print("기존 페르소나를 재사용합니다.")
        return {"target_personas": state['target_personas']}

    # Pydantic 모델을 사용하는 JSON 파서 초기화
    pydantic_parser = JsonOutputParser(pydantic_object=Personas)

    input_data = state.get('input_data', {})
    core_benefit_text = input_data.get('coreBenefitText', '기본 혜택')
    refine_feedback = state.get('refine_feedback', None)
    custom_columns = input_data.get('customColumns', {})

    if isinstance(custom_columns, dict):
        formatted_columns = "\n".join([f"- {k}: {v}" for k, v in custom_columns.items()])
    else:
        formatted_columns = str(custom_columns)

    # 소비자 세그먼트 논문 기반 RAG
    segment_knowledge = rag_search_targeting(
        query="소비자 세그먼트 분류 기준 및 소비 패턴별 그룹 특징"
    )

    # RAG Tool 호출: 정책 관련 지식 검색
    policy_knowledge = rag_search(query=f"{core_benefit_text} 관련 정책", source_type='정책')
    print(f"Targeting Agent - RAG Knowledge: {policy_knowledge}")

    # LLM 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 KT의 전문 마케팅 분석가입니다. 
            아래 세 가지 정보를 기반으로 5개의 서로 다른 타겟 세그먼트를 도출해야 합니다:

            1) 프로모션 핵심 혜택  
            2) 마케터가 제공한 customColumns (고객 DB의 Feature)  
            3) 소비자 세그먼트 논문 기반 RAG 지식  

            **중요 규칙**
            - 각 세그먼트는 반드시 customColumns 중 최소 1개 이상을 기반으로 해야 합니다.
            - 논문 기반 소비 패턴 / 세그먼트 기준을 반드시 반영해야 합니다.
            - 현실적인 고객 DB 세그멘테이션 규칙(구매 빈도, 나이, 선호 카테고리 등)을 반영해야 합니다.
            - 단순 페르소나가 아니라 **데이터 기반 세그먼트 그룹**을 출력해야 합니다.
            - 모든 출력 필드(target_name, target_features, classification_reason)는 반드시 한국어로 작성해야 합니다.
            - 마케터 수정 피드백은 세그먼트 내용을 구성하는 데에만 참고하고, JSON 출력 형식은 반드시 유지해야 합니다.

            {format_instructions}
            """),

            ("human", """
            프로모션 핵심 혜택:
            {core_benefit}

            마케터 수정 피드백:
            {refine_feedback_text}

            사용 가능한 고객 데이터 컬럼(customColumns):
            {custom_columns}

            소비자 세그먼트 관련 RAG 지식:
            {segment_knowledge}

            프로모션 정책 관련 RAG 지식:
            {policy_knowledge}

            위 정보를 기반으로 5개의 데이터 기반 타겟 세그먼트를 생성해주세요.
            """)
        ]).partial(format_instructions=pydantic_parser.get_format_instructions())

    # LangChain Expression Language (LCEL) 체인 구성
    chain = prompt | llm | pydantic_parser

    # 체인 실행
    response_dict = chain.invoke({
        "core_benefit": core_benefit_text,
        "refine_feedback_text": refine_feedback.get('details', '없음') if refine_feedback else '없음',
        "custom_columns": formatted_columns,
        "segment_knowledge": segment_knowledge,
        "policy_knowledge": policy_knowledge
    })

    # Pydantic 파서는 이미 딕셔너리를 반환합니다.
    target_personas = response_dict.get("personas", [])
    print(f"Targeting Agent - 생성된 타겟 페르소나: {target_personas}")
    return {"target_personas": target_personas}

def run_messaging_agent(state: CampaignState) -> Dict:
    print("--- Messaging Agent 실행 중 ---")
    # 상태에서 필요한 데이터 추출
    input_data = state.get('input_data', {})
    target_personas = state.get('target_personas', [])
    rework_count = state.get('rework_count', 0)
    validation_reports = state.get('validation_reports')
    refine_feedback = state.get('refine_feedback', None)

    # 공통으로 사용될 데이터 구성
    core_benefit_text = input_data.get('coreBenefitText', '기본 혜택')
    custom_columns_data = input_data.get('customColumns', {})
    source_urls = input_data.get('sourceUrls', [])
    source_urls_str = ", ".join(source_urls) if source_urls else '없음'

    if isinstance(custom_columns_data, dict):
        columns_list = [f"- `{{{k}}}`: ({v})" for k, v in custom_columns_data.items()]
        columns_for_prompt = "\n".join(columns_list)
    else:
        columns_for_prompt = ", ".join(custom_columns_data)

    # 새로운 프롬프트: 단계적 사고(Chain-of-Thought)와 듀얼 RAG 적용
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
        당신은 고객의 감정을 움직이는 초개인화 마케팅 메시지 전문 카피라이터입니다.
        아래의 3단계 프로세스를 엄격히 따라서, 주어진 타겟 페르소나를 위한 메시지 초안 2개를 생성해야 합니다.

        ---
        **[1단계: 분석 및 전략 수립]**

        먼저, 주어진 모든 정보(페르소나, 핵심 혜택, RAG 지식)를 종합적으로 분석하고, 각 초안에 대한 생성 전략을 머릿속으로 구체적으로 수립합니다.
        아래 <생각 예시>는 당신의 사고 과정을 돕기 위한 참고 자료일 뿐, **이 내용을 그대로 모방하거나 실제 생성 메시지에 사용해서는 안 됩니다.**

        <생각 예시>
        1.  **페르소나 분석**: 타겟은 '20대 기술에 민감한 대학생'. 가격에 민감하지만 최신 기술 경험을 중시함.
        2.  **RAG 지식 분석**: 성공 사례를 보니, 이 그룹은 명확한 숫자 비교와 직접적인 화법에 반응이 좋음. 실패 사례에서는 유치한 이모티콘과 전문 용어 남발을 싫어하는 경향이 나타남.
        3.  **초안 1 (실속형) 전략**: '50% 할인'과 '2배 빠른 속도'라는 혜택의 숫자를 전면에 내세운다. 과장된 표현 없이 간결하고 직설적인 톤으로 작성한다.
        4.  **초안 2 (감정형) 전략**: '남들보다 앞서가는 경험', '친구들 사이에서 돋보이는 최신 기기'라는 자부심을 자극한다. 좀 더 트렌디하고 세련된 톤으로 작성한다.
        </생각 예시>

        ---
        **[2단계: 메시지 초안 작성]**

        위에서 수립한 전략에 따라, 아래 규칙을 준수하여 메시지 초안 2개를 작성합니다.

        *   **핵심 혜택 반영**: `<coreBenefitText>` 안의 모든 내용을 어떤 항목도 생략/삭제/변경 없이 본문에 자연스럽게 포함해야 합니다.
        *   **메시지 구조**: [오프닝] - [본문] - [프로모션 기간] - [CTA] 순서를 따릅니다.
        *   **피드백 반영**: 수정 피드백이 있다면, 반드시 해당 내용을 반영하여 작성합니다.
        *   **초안별 규칙**:

            -   **[초안 1: 실속형 메시지]**
                -   **목적**: 고객이 얻는 금전적, 기능적 이득을 명확히 인지시키는 것.
                -   **핵심 규칙**:
                    -   첫 문장은 반드시 할인율, 포인트, 금액 등 **숫자로 표현된 혜택**으로 시작해야 합니다.
                    -   본문에는 '~만', '~부터', '~까지' 등 **범위나 한정을 나타내는 표현**을 사용하여 혜택의 구체성을 더하세요.
                    -   감성적이거나 추상적인 표현(예: '특별한 경험', '놀라운')은 **절대 사용하지 마세요.**

            -   **[초안 2: 감정형 메시지]**
                -   **목적**: 해당 상품/서비스가 고객의 삶에 가져올 긍정적인 감정이나 변화를 그려주는 것.
                -   **핵심 규칙**:
                    -   첫 문장은 반드시 고객의 상황이나 감정에 공감하는 **질문**으로 시작해야 합니다. (예: "요즘 부쩍 지쳐 보인다는 말을 듣지 않으셨나요?")
                    -   '선물', '나를 위한', '당신만의' 등 **개인화되고 감성적인 키워드**를 2개 이상 사용하세요.
                    -   할인율, 포인트 등 **숫자로 된 혜택을 직접적으로 언급하는 것을 피하세요.** (혜택 자체는 설명하되, 숫자는 제외)

        *   **[!!!]** 위 규칙에 따라 두 초안의 내용과 구조는 눈에 띄게 달라야 하며, 서로 조금이라도 비슷하게 작성될 경우 생성은 실패한 것으로 간주합니다.

        ---
        **[3단계: 최종 출력]**

        생성된 메시지를 반드시 아래 JSON 형식에 맞춰 최종 출력합니다. 그 어떤 추가 설명도 붙이지 마세요.
        {{
        "drafts": [
            {{
                "message_draft_index": 1,
                "message_text": "(실속형으로 작성된 전체 메시지 텍스트)"
            }},
            {{
                "message_draft_index": 2,
                "message_text": "(감정형으로 작성된 전체 메시지 텍스트)"
            }}
        ]
        }}
        """
         )
    ])
    chain = prompt | llm | json_parser

    # 헬퍼 함수: RAG 검색 및 포맷팅
    def get_rag_knowledge_for_persona(target_name: str) -> str:
        success_query = f"{target_name} 타겟 메시지 성공 사례"
        failure_query = f"{target_name} 타겟 메시지 실패 사례"
        
        success_knowledge = rag_search(query=success_query, source_type='성공 사례')
        failure_knowledge = rag_search(query=failure_query, source_type='실패 사례')
        
        return f"[참고할 성공 사례]\n{success_knowledge}\n\n[피해야 할 실패 사례]\n{failure_knowledge}"

    # 시나리오 1: 마케터의 refine 요청 처리 (항상 전체 재작성)
    if refine_feedback:
        print("--- 실행 모드: 마케터 피드백 기반 전체 재작업 ---")
        final_drafts = []
        for persona in target_personas:
            feedback_instructions = "아래 마케터 피드백을 반영해 수정하여 작성하세요."
            feedback_section = f"마케터 피드백: {refine_feedback.get('details', '없음')}"
            rag_knowledge = get_rag_knowledge_for_persona(persona['target_name'])
            
            response = chain.invoke({
                "feedback_instructions": feedback_instructions, "feedback_section": feedback_section,
                "target_name": persona['target_name'], "target_features": persona['target_features'],
                "core_benefit": core_benefit_text, "columns": columns_for_prompt, "source_urls": source_urls_str,
                "rag_knowledge": rag_knowledge
            })
            final_drafts.append({
                "target_group_index": persona['target_group_index'], "target_name": persona['target_name'],
                "message_drafts": response.get("drafts", [])
            })
        return {"messages_drafts": final_drafts, "rework_count": 0}

    # 시나리오 2 & 3: 검증 결과에 따른 재작업 또는 초기 생성
    is_rework = False
    personas_to_rework = set()
    feedback_per_persona = {}

    if validation_reports:
        for report in validation_reports:
            if report.get('policy_compliance') == 'FAIL' or report.get('spam_risk_score', 0) > 70:
                is_rework = True
                group_index = report['target_group_index']
                draft_index = report['message_draft_index']
                feedback = report.get('recommended_action', '피드백 없음')
                
                personas_to_rework.add(group_index)
                if group_index not in feedback_per_persona:
                    feedback_per_persona[group_index] = []
                feedback_per_persona[group_index].append(f"초안 {draft_index}: {feedback}")

    if not is_rework:
        # --- 시나리오 2: 초기 생성 ---
        print("--- 실행 모드: 초기 생성 ---")
        initial_drafts = []
        for persona in target_personas:
            rag_knowledge = get_rag_knowledge_for_persona(persona['target_name'])
            response = chain.invoke({
                "feedback_instructions": "", "feedback_section": "",
                "target_name": persona['target_name'], "target_features": persona['target_features'],
                "core_benefit": core_benefit_text, "columns": columns_for_prompt, "source_urls": source_urls_str,
                "rag_knowledge": rag_knowledge
            })
            initial_drafts.append({
                "target_group_index": persona['target_group_index'], "target_name": persona['target_name'],
                "message_drafts": response.get("drafts", [])
            })
        return {"messages_drafts": initial_drafts, "rework_count": rework_count}
    else:
        # --- 시나리오 3: 부분 재작업 ---
        print(f"--- 실행 모드: 부분 재작업 (대상 페르소나: {list(personas_to_rework)}) ---")
        previous_drafts = state.get('messages_drafts', [])
        final_drafts = []
        
        persona_map = {p['target_group_index']: p for p in target_personas}
        previous_drafts_map = {d['target_group_index']: d for d in previous_drafts}

        for group_index in sorted(persona_map.keys()):
            persona = persona_map[group_index]
            
            if group_index in personas_to_rework:
                print(f"재작업 실행: 타겟 그룹 {group_index} (시도 횟수: {rework_count + 1})")
                all_feedback_for_persona = "\n".join(feedback_per_persona[group_index])
                
                # 재작업 횟수에 따라 피드백 지시사항 강화
                if rework_count > 0:
                    feedback_instructions = "이전 수정 요청이 제대로 반영되지 않았습니다. 아래 피드백을 **반드시 엄격하게 준수하여** 메시지를 **전면적으로 재작성**하세요."
                else:
                    feedback_instructions = "아래 수정 피드백을 반영해 메시지를 다시 작성하세요."

                feedback_section = f"수정 피드백:\n{all_feedback_for_persona}"
                rag_knowledge = get_rag_knowledge_for_persona(persona['target_name'])
                
                response = chain.invoke({
                    "feedback_instructions": feedback_instructions, "feedback_section": feedback_section,
                    "target_name": persona['target_name'], "target_features": persona['target_features'],
                    "core_benefit": core_benefit_text, "columns": columns_for_prompt, "source_urls": source_urls_str,
                    "rag_knowledge": rag_knowledge
                })
                final_drafts.append({
                    "target_group_index": group_index, "target_name": persona['target_name'],
                    "message_drafts": response.get("drafts", [])
                })
            else:
                print(f"초안 유지: 타겟 그룹 {group_index}")
                if group_index in previous_drafts_map:
                    final_drafts.append(previous_drafts_map[group_index])
        
        return {"messages_drafts": final_drafts, "rework_count": rework_count + 1}

def run_validator_agent(state: CampaignState) -> Dict:
    """
    Validator Agent: 생성된 메시지 초안을 검증하고, 필요한 경우 피드백을 제공합니다.
    """
    print("---" + " Validator Agent 실행 중 ---")
    messages_drafts = state.get('messages_drafts', [])
    core_benefit_text = state.get('input_data', {}).get('coreBenefitText', '')

    # LLM 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 메시지 검토 및 법규 준수 전문가입니다. 당신의 임무는 주어진 메시지 초안을 아래 3가지 관점에서
        **엄격하게 평가**하고 구조화된 JSON 리포트를 작성하는 것입니다.

        1.  **스팸 위험도 (0~100점):** 과도한 이모티콘, 특수문자, 긴급성 강조 문구 사용 여부. 점수가 높을수록 위험.
        2.  **정보의 정확성/정책 준수:** RAG 지식 기반으로 혜택 조건 등이 사실과 일치하는지 확인.
        3.  **개선 의견:** 실제 발송 전 수정이 필요한 부분을 명확히 제시.

        결과는 반드시 아래 JSON 형식의 단일 객체로 반환해야 합니다.
        'policy_compliance'가 'FAIL'일 경우, 'review_summary'는 반드시 "위반 사유: [인용문]" 으로 시작해야 하며, RAG 지식에서 위반된 정책의 핵심 내용을 정확히 인용해야 합니다.
        {{
            "spam_risk_score": <0-100 사이의 정수>,
            "policy_compliance": "<'PASS' 또는 'FAIL'>",
            "review_summary": "<(FAIL 시) 위반 사유: [인용문]을 포함한 검토 요약>",
            "recommended_action": "<구체적인 개선 제안 또는 '없음'>"
        }}
        """),
        ("human", """
        검토할 메시지 초안:
        ---
        {message_text}
        ---
        
        프로모션 핵심 혜택: {core_benefit}
        참고용 RAG 지식 (스팸/광고 정책): {rag_knowledge}

        위 정보를 바탕으로 메시지 초안을 평가하고 JSON 리포트를 작성해주세요.
        """)
    ])

    # LangChain Expression Language (LCEL) 체인 구성
    chain = prompt | llm | json_parser

    validation_reports = []

    # RAG Tool 호출: 스팸/광고 정책을 한 번만 검색
    spam_policy_knowledge = rag_search(query="메시지 스팸/광고 정책", source_type='스팸/광고 정책')
    print(f"Validator Agent - RAG Knowledge for validation: {spam_policy_knowledge}")

    for target_group_drafts in messages_drafts:
        target_name = target_group_drafts['target_name']
        for draft in target_group_drafts['message_drafts']:
            message_text = draft['message_text']

            # 체인 실행
            report = chain.invoke({
                "message_text": message_text,
                "core_benefit": core_benefit_text,
                "rag_knowledge": spam_policy_knowledge
            })

            # 전체 리포트 저장
            report['target_group_index'] = target_group_drafts['target_group_index']
            report['message_draft_index'] = draft['message_draft_index']
            validation_reports.append(report)

    print(f"Validator Agent - 생성된 검증 리포트: {validation_reports}")

    # 에이전트는 이제 리포트만 반환하고, 재작업 결정은 decide_next_step에서 처리합니다.
    # 이전 피드백 상태를 확실히 지우기 위해 validator_feedback을 None으로 설정합니다.
    return {"validation_reports": validation_reports, "validator_feedback": None}


def run_formatter_agent(state: CampaignState) -> Dict:
    """
    Formatter Agent: 최종 결과를 통합하여 BE 서버로 전달할 JSON 형태로 포맷팅합니다.
    이 버전에서는 타겟 페르소나, 메시지 초안, 검증 리포트를 모두 결합합니다.
    """
    print("---" + " Formatter Agent 실행 중 ---")
    target_personas = state.get('target_personas', [])
    messages_drafts = state.get('messages_drafts', [])
    validation_reports = state.get('validation_reports', [])

    # 빠른 조회를 위해 리포트와 초안을 맵으로 변환합니다.
    report_map = {}
    if validation_reports:
        for report in validation_reports:
            key = (report['target_group_index'], report['message_draft_index'])
            report_map[key] = report

    draft_map = {}
    if messages_drafts:
        for group in messages_drafts:
            draft_map[group['target_group_index']] = group['message_drafts']

    # 페르소나를 기준으로 초안과 검증 리포트를 결합합니다.
    final_target_groups = []
    if target_personas:
        for persona in target_personas:
            group_index = persona['target_group_index']
            drafts_for_group = draft_map.get(group_index, [])
            
            new_drafts = []
            for draft in drafts_for_group:
                key = (group_index, draft['message_draft_index'])
                report_for_draft = report_map.get(key)
                
                new_draft_entry = {
                    "message_draft_index": draft['message_draft_index'],
                    "message_text": draft['message_text'],
                    "validation_report": report_for_draft
                }
                new_drafts.append(new_draft_entry)
            
            final_target_groups.append({
                "target_group_index": group_index,
                "target_name": persona['target_name'],
                "target_features": persona['target_features'],
                "classification_reason": persona.get('classification_reason', 'N/A'), # 이유 필드 추가
                "message_drafts": new_drafts
            })

    print(f"Formatter Agent - 최종 결합 결과: {final_target_groups}")
    return {"final_output": final_target_groups}

# 3. LangGraph 조건부 루프: decide_next_step 함수
def decide_next_step(state: CampaignState) -> str:
    """
    Validator 노드 이후 다음 단계를 결정합니다.
    재시도 횟수 및 검증 결과에 따라 'messaging' 노드로 루프백하거나 'formatter' 노드로 종료됩니다.
    """
    print("---" + " decide_next_step 실행 중 ---")
    rework_count = state.get('rework_count', 0)
    validation_reports = state.get('validation_reports', [])

    # 최대 재시도 횟수 (2회) 초과 시 강제 종료
    if rework_count >= 2:
        print(f"재시도 횟수 {rework_count}회 초과. Formatter로 이동하여 강제 종료.")
        return "formatter"

    # validation_reports를 직접 검사하여 재작업 필요 여부 확인
    needs_rework = False
    if validation_reports:
        for report in validation_reports:
            if report.get('policy_compliance') == 'FAIL' or report.get('spam_risk_score', 0) > 70:
                needs_rework = True
                break  # 하나라도 실패하면 즉시 재작업 결정

    if needs_rework:
        print(f"검증 실패. Messaging Agent로 루프백하여 메시지 재생성 시도. 현재 재시도 횟수: {rework_count}")
        return "messaging"
    else:
        print("모든 검증 통과. Formatter로 이동하여 최종 결과 포맷팅.")
        return "formatter"

# LangGraph 워크플로우 빌드
def build_agent_workflow():
    workflow = StateGraph(CampaignState)

    # 노드 추가
    workflow.add_node("targeting", run_targeting_agent)
    workflow.add_node("messaging", run_messaging_agent)
    workflow.add_node("validator", run_validator_agent)
    workflow.add_node("formatter", run_formatter_agent)

    # 시작점 설정 (분기 가능하도록)
    # 기본 시작점은 'targeting'
    workflow.set_entry_point("targeting") 
    # 'messaging'을 또 다른 진입점으로 설정
    # workflow.add_entry_point("messaging") # Removed as it causes an error

    # 엣지 연결
    workflow.add_edge("targeting", "messaging")
    workflow.add_edge("messaging", "validator")

    # 조건부 엣지 연결
    workflow.add_conditional_edges(
        "validator",
        decide_next_step,
        {
            "messaging": "messaging", # 재작업 필요 시 messaging 노드로 루프백
            "formatter": "formatter"  # 검증 성공 또는 재시도 횟수 초과 시 formatter 노드로
        }
    )

    # 종료 엣지
    workflow.add_edge("formatter", END)

    app = workflow.compile(checkpointer=None)
    return app

# 워크플로우 테스트 (선택 사항)
if __name__ == "__main__":
    app = build_agent_workflow()

    initial_state = {
        "input_data": {
            "core_benefit_text": "KT 5G 프리미엄 요금제, 데이터 완전 무제한!",
            "message_tone": "전문적이고 친근한",
            "custom_columns": ["[이름]", "[핸드폰기종]", "[사용년도]"]
        },
        "rework_count": 0,
        "target_personas": None,
        "messages_drafts": None,
        "validation_reports": None,
        "validator_feedback": None,
        "refine_feedback": None
    }

    print("---" + " LangGraph 워크플로우 시작 ---")
    # 스트리밍 방식으로 실행 결과를 확인합니다.
    for s in app.stream(initial_state):
        print(s)
        print("---")
    print("---" + " LangGraph 워크플로우 종료 ---")
