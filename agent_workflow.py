import os
from typing import List, Dict, TypedDict, Union
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from rag_utils import query_chroma

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
    validator_feedback: Union[Dict, None] # Validator가 Messaging Agent에게 전달할 구체적인 수정 피드백
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

# 2. Agent 함수 구현 (LLM 연동)

# LLM, Parser, Prompt 등 공통 컴포넌트 초기화
# 참고: OpenAI API 키는 환경변수 'OPENAI_API_KEY'에 설정되어 있어야 합니다.
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
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

    input_data = state.get('input_data', {})
    core_benefit_text = input_data.get('coreBenefitText', '기본 혜택')
    refine_feedback = state.get('refine_feedback', None)

    # RAG Tool 호출: 정책 관련 지식 검색
    policy_knowledge = rag_search(query=f"{core_benefit_text} 관련 정책", source_type='정책')
    print(f"Targeting Agent - RAG Knowledge: {policy_knowledge}")

    # LLM 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 KT의 전문 마케팅 분석가입니다. 당신의 임무는 주어진 프로모션의 핵심 혜택과 관련 지식을 바탕으로,
        이 혜택을 가장 매력적으로 느낄 **5개의 서로 다른 가상 타겟 페르소나**를 도출하는 것입니다. 
        만약 마케터의 수정 피드백이 있다면, 이를 최우선으로 고려하여 페르소나를 수정하거나 재생성해야 합니다.

        각 페르소나는 다음을 포함해야 합니다:
        - `target_group_index`: 1부터 5까지의 순서 번호.
        - `target_name`: 페르소나를 대표하는 이름 (예: '20대 초반 대학생', '30대 직장인').
        - `target_features`: 페르소나의 상세 특징, 니즈, 라이프스타일.

        결과는 반드시 아래 JSON 형식의 단일 객체로 반환해야 합니다.
        {{
            "personas": [
                {{
                    "target_group_index": 1,
                    "target_name": "...",
                    "target_features": "..."
                }},
                ... 4 more personas
            ]
        }}
        """),
        ("human", """
        프로모션 핵심 혜택: {core_benefit}
        마케터 수정 피드백: {refine_feedback_text}
        관련 지식 (RAG): {rag_knowledge}

        위 정보를 바탕으로 5개의 타겟 페르소나를 생성해주세요.
        """)
    ])

    # LangChain Expression Language (LCEL) 체인 구성
    chain = prompt | llm | json_parser

    # 체인 실행
    response = chain.invoke({
        "core_benefit": core_benefit_text,
        "refine_feedback_text": refine_feedback.get('details', '없음') if refine_feedback else '없음',
        "rag_knowledge": policy_knowledge
    })
    
    target_personas = response.get("personas", [])
    print(f"Targeting Agent - 생성된 타겟 페르소나: {target_personas}")
    return {"target_personas": target_personas}

def run_messaging_agent(state: CampaignState) -> Dict:
    """
    Messaging Agent: 각 타겟 페르소나에 맞춰 초개인화 메시지 초안 2개를 생성합니다.
    Validator Agent로부터의 피드백을 반영하여 메시지를 재생성할 수 있습니다.
    """
    print("---" + " Messaging Agent 실행 중 ---")
    input_data = state.get('input_data', {})
    target_personas = state.get('target_personas', [])
    rework_count = state.get('rework_count', 0)
    validator_feedback = state.get('validator_feedback', None)
    refine_feedback = state.get('refine_feedback', None) # 마케터 피드백 추가
    core_benefit_text = input_data.get('core_benefit_text', '기본 혜택')
    custom_columns = input_data.get('custom_columns', ['[이름]', '[핸드폰기종]', '[사용년도]'])

    # LLM 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 고객의 마음을 사로잡는 초개인화 마케팅 메시지 카피라이터입니다.
        주어진 타겟 페르소나의 특징을 명확히 언급하고, 제공된 커스텀 컬럼을 **반드시 1개 이상 활용**하여
        고객이 '나만을 위한 메시지'라고 느끼도록 **2개의 서로 다른 초안**을 작성하십시오.
        참고용 RAG 지식(성공 사례)을 활용하여 효과적인 문구를 구성하세요.

        {feedback_instructions}

        결과는 반드시 아래 JSON 형식의 단일 객체로 반환해야 합니다.
        {{
            "drafts": [
                {{
                    "message_draft_index": 1,
                    "message_text": "[타겟 그룹명]\\n\\n(메시지 내용)"
                }},
                {{
                    "message_draft_index": 2,
                    "message_text": "[타겟 그룹명]\\n\\n(메시지 내용)"
                }}
            ]
        }}
        """),
        ("human", """
        타겟 페르소나 이름: {target_name}
        타겟 페르소나 특징: {target_features}
        프로모션 핵심 혜택: {core_benefit}
        사용 가능한 커스텀 컬럼: {columns}
        참고용 RAG 지식 (성공 사례): {rag_knowledge}
        {feedback_section}

        위 정보를 바탕으로 2개의 메시지 초안을 생성해주세요.
        """)
    ])

    # LangChain Expression Language (LCEL) 체인 구성
    chain = prompt | llm | json_parser

    messages_drafts = []
    for persona in target_personas:
        target_name = persona['target_name']
        target_features = persona['target_features']

        # RAG Tool 호출: 성공 사례 검색
        success_case_knowledge = rag_search(query=f"{target_name} 타겟 메시지 성공 사례", source_type='성공 사례')
        print(f"Messaging Agent - RAG Knowledge for {target_name}: {success_case_knowledge}")

        # 피드백 처리
        feedback_instructions = ""
        feedback_section = ""
        if refine_feedback: # 마케터의 피드백을 최우선으로 반영
            print(f"Messaging Agent - Marketer's Refine Feedback 반영 중: {refine_feedback}")
            feedback_instructions = "이전 초안에 대한 아래의 마케터 피드백을 반영하여 메시지를 수정해주세요."
            feedback_section = f"마케터 수정 피드백: {refine_feedback.get('details', '없음')}"
        elif validator_feedback: # 마케터 피드백이 없을 경우, Validator 피드백 반영
            print(f"Messaging Agent - Validator Feedback 반영 중: {validator_feedback}")
            feedback_instructions = "이전 초안에 대한 아래의 피드백을 반영하여 메시지를 수정해주세요."
            feedback_section = f"수정 피드백: {validator_feedback.get('details', '없음')}"
            

        # 체인 실행
        response = chain.invoke({
            "feedback_instructions": feedback_instructions,
            "target_name": target_name,
            "target_features": target_features,
            "core_benefit": core_benefit_text,
            "columns": ", ".join(custom_columns),
            "rag_knowledge": success_case_knowledge,
            "feedback_section": feedback_section
        })

        # 생성된 초안을 persona 정보와 함께 저장
        messages_drafts.append({
            "target_group_index": persona['target_group_index'],
            "target_name": target_name,
            "message_drafts": response.get("drafts", [])
        })
        
    print(f"Messaging Agent - 생성된 메시지 초안: {messages_drafts}")
    # validator_feedback이 있었다면 rework_count를 1 증가시킴
    return {"messages_drafts": messages_drafts, "rework_count": rework_count + 1 if validator_feedback else rework_count}

def run_validator_agent(state: CampaignState) -> Dict:
    """
    Validator Agent: 생성된 메시지 초안을 검증하고, 필요한 경우 피드백을 제공합니다.
    """
    print("---" + " Validator Agent 실행 중 ---")
    messages_drafts = state.get('messages_drafts', [])
    core_benefit_text = state.get('input_data', {}).get('core_benefit_text', '')

    # LLM 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 메시지 검토 및 법규 준수 전문가입니다. 당신의 임무는 주어진 메시지 초안을 아래 3가지 관점에서
        **엄격하게 평가**하고 구조화된 JSON 리포트를 작성하는 것입니다.

        1.  **스팸 위험도 (0~100점):** 과도한 이모티콘, 특수문자, 긴급성 강조 문구 사용 여부. 점수가 높을수록 위험.
        2.  **정보의 정확성/정책 준수:** RAG 지식 기반으로 혜택 조건 등이 사실과 일치하는지 확인.
        3.  **개선 의견:** 실제 발송 전 수정이 필요한 부분을 명확히 제시.

        결과는 반드시 아래 JSON 형식의 단일 객체로 반환해야 합니다.
        {{
            "spam_risk_score": <0-100 사이의 정수>,
            "policy_compliance": "<'PASS' 또는 'FAIL'>",
            "review_summary": "<검토 요약>",
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
    needs_rework = False
    validator_feedback = {"reason": "초안 메시지 검증 결과, 수정이 필요합니다.", "details": []}

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

            # 검증 실패 조건 확인
            if report.get('policy_compliance') == 'FAIL' or report.get('spam_risk_score', 0) > 70:
                needs_rework = True
                feedback_detail = (
                    f"타겟 '{target_name}'의 메시지 초안 {draft['message_draft_index']}: "
                    f"{report.get('recommended_action', '피드백 없음')}"
                )
                validator_feedback['details'].append(feedback_detail)

            # 전체 리포트 저장
            report['target_group_index'] = target_group_drafts['target_group_index']
            report['message_draft_index'] = draft['message_draft_index']
            validation_reports.append(report)

    print(f"Validator Agent - 생성된 검증 리포트: {validation_reports}")

    if needs_rework:
        # 피드백의 details를 하나의 문자열로 합침
        feedback_str = "\n".join(validator_feedback['details'])
        return {"validation_reports": validation_reports, "validator_feedback": {"details": feedback_str}}
    else:
        return {"validation_reports": validation_reports, "validator_feedback": None}


def run_formatter_agent(state: CampaignState) -> Dict:
    """
    Formatter Agent: 최종 결과를 통합하여 BE 서버로 전달할 JSON 형태로 포맷팅합니다.
    """
    print("---" + " Formatter Agent 실행 중 ---")
    # BE API 명세에 맞춰 'messages_drafts'를 최종 결과로 포맷팅합니다.
    # 이것이 BE가 기대하는 'target_groups' 리스트가 됩니다.
    final_result = state.get('messages_drafts')
    
    print(f"Formatter Agent - 최종 결과 (messages_drafts): {final_result}")
    return {"final_output": final_result} # 최종 결과는 'final_output' 키로 반환

# 3. LangGraph 조건부 루프: decide_next_step 함수
def decide_next_step(state: CampaignState) -> str:
    """
    Validator 노드 이후 다음 단계를 결정합니다.
    재시도 횟수 및 검증 결과에 따라 'messaging' 노드로 루프백하거나 'formatter' 노드로 종료됩니다.
    """
    print("---" + " decide_next_step 실행 중 ---")
    rework_count = state.get('rework_count', 0)
    validation_reports = state.get('validation_reports', [])
    validator_feedback = state.get('validator_feedback', None)

    # 최대 재시도 횟수 (2회) 초과 시 강제 종료
    if rework_count >= 2:
        print(f"재시도 횟수 {rework_count}회 초과. Formatter로 이동하여 강제 종료.")
        return "formatter"

    # 검증 실패 조건 확인 (예: policy_compliance == 'FAIL' 또는 스팸 점수 기준 초과)
    # 하나라도 FAIL이거나 스팸 점수가 높으면 재작업 필요
    needs_rework = False
    if validator_feedback and validator_feedback.get('details'):
        needs_rework = True

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
