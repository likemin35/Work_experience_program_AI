# Message AI Server

메시지 생성 프로젝트의 AI 서버 저장소입니다. 이 서버는 프로모션 PDF에서 핵심 정보를 추출하고, 고객 CSV 기반 세그먼트를 생성하며, 세그먼트별 맞춤형 메시지 초안을 만듭니다.

특히 현재 버전에는 단순 생성 API를 넘어서, 과거 프로모션 이력과 성과 데이터를 참고해 전략을 먼저 고르는 `Strategy Agent` 구조가 포함되어 있습니다.

## 서비스 개요

- 역할: AI 전용 백엔드 서비스
- 배포 환경: Azure Container Apps
- 프론트 서비스 주소: https://message-fe-app.redriver-ce1c37ed.japaneast.azurecontainerapps.io/
- 별도 MCP 연동 플랫폼 주소: https://marketing-platform-app.redriver-ce1c37ed.japaneast.azurecontainerapps.io/

## 주요 기능

- 프로모션 PDF 텍스트 추출
- OpenAI 기반 프로모션 정보 구조화
- 고객 설명 기반 세그먼트 클러스터링
- 세그먼트별 개인화 메시지 초안 생성
- 과거 프로모션 이력 기반 전략 선택
- MCP / 외부 플랫폼 / Notion 기반 이력 데이터 소스 확장

## 핵심 구조

### 1. PDF 추출

- `POST /ai/campaign/extract`
- PDF 파일을 입력받아 제목, 핵심 혜택, 프로모션 URL을 추출합니다.
- 관련 파일: `upload_pdf.py`

### 2. 세그먼트 생성

- `POST /cluster-customers`
- 캠페인 정보와 고객 설명 목록을 입력받아 고객 그룹을 만듭니다.
- 이 과정에서 `Strategy Agent`가 먼저 과거 유사 프로모션을 조회한 뒤,
  `reuse`, `hybrid`, `new` 중 어떤 방식으로 세그먼트를 나눌지 결정합니다.
- 관련 파일: `app.py`, `strategy_agent.py`

### 3. 메시지 생성

- `POST /generate-messages`
- 세그먼트 특징을 입력받아 CRM 메시지 초안을 생성합니다.
- 각 세그먼트마다 `Strategy Agent`가 과거 메시지 패턴을 참고할지 판단합니다.
- 관련 파일: `create_message.py`, `strategy_agent.py`

## Strategy Agent 설명

이 저장소의 Agent 기능은 단순 챗봇이 아니라, 생성 전에 전략을 고르는 오케스트레이션 레이어입니다.

### Agent가 하는 일

1. 현재 캠페인 정보와 세그먼트 후보를 받습니다.
2. History Provider를 통해 유사한 과거 프로모션을 조회합니다.
3. 과거 성과를 기준으로 참고할 이력을 정렬합니다.
4. LLM에게 현재 케이스에서 어떤 전략이 적절한지 판단하게 합니다.
5. 그 결과를 실제 세그먼트 생성 프롬프트 또는 메시지 생성 프롬프트에 반영합니다.

### 선택 가능한 전략 모드

- `reuse`: 과거 전략을 거의 그대로 활용
- `hybrid`: 과거 성과가 좋은 패턴을 참고하되 현재 데이터에 맞게 조정
- `new`: 과거 이력 대신 현재 입력 기준으로 새 전략 생성

### 세그먼트 전략 결정

- 메서드: `CampaignStrategyAgent.decide_clustering_strategy()`
- 입력:
  - 캠페인 제목
  - 캠페인 혜택 설명
  - 고객 설명 목록
- 출력:
  - `strategy_mode`
  - `reason`
  - `segmentation_guidance`
  - `history_references`

### 메시지 전략 결정

- 메서드: `CampaignStrategyAgent.decide_message_strategy()`
- 입력:
  - 캠페인 정보
  - 세그먼트 이름
  - 세그먼트 특징
- 출력:
  - `message_mode`
  - `reason`
  - `message_guidance`
  - `history_references`

## History Provider 설명

Agent는 직접 DB를 읽지 않고 `history_provider.py`를 통해 과거 프로모션 이력을 가져옵니다. 이 계층 덕분에 데이터 소스를 바꿔도 Agent 로직은 그대로 유지됩니다.

### 지원 소스

- `mock`
  - 샘플 데이터를 반환하는 안전한 기본 모드
- `web_platform`
  - 별도 웹 플랫폼 API를 조회
- `mcp`
  - MCP 게이트웨이를 통해 검색 API 호출
- `notion_api`
  - Notion Database를 직접 조회

### 동작 방식

1. `build_history_provider()`가 환경 변수 기반으로 provider를 선택합니다.
2. provider가 유사 프로모션 목록을 반환합니다.
3. Agent가 성과 지표를 기준으로 정렬합니다.
4. 상위 이력을 LLM 판단에 전달합니다.

### 성과 정렬 기준

아래 가중치를 이용해 과거 프로모션 점수를 계산합니다.

- 클릭률: `WEIGHT_CLICK_RATE`
- 참여율: `WEIGHT_PARTICIPATION_RATE`
- 전환율: `WEIGHT_CONVERSION_RATE`

기본값은 다음과 같습니다.

- `WEIGHT_CLICK_RATE=0.5`
- `WEIGHT_PARTICIPATION_RATE=0.3`
- `WEIGHT_CONVERSION_RATE=0.2`

## API 응답에서 Agent 결과가 쓰이는 위치

### 세그먼트 생성 응답

`/cluster-customers` 응답에는 생성된 `clusters`와 함께 `strategy_meta`가 포함됩니다.

예시 구조:

```json
{
  "clusters": [],
  "strategy_meta": {
    "strategy_mode": "hybrid",
    "reason": "Historical campaigns for similar benefits performed well with value-based grouping.",
    "segmentation_guidance": "Keep group styles close to high-performing past campaigns.",
    "history_references": ["promotion-101", "promotion-102"]
  }
}
```

### 메시지 생성 응답

`/generate-messages` 응답의 각 세그먼트 결과 안에는 `strategy_meta`가 포함됩니다.

예시 구조:

```json
{
  "title": "campaign title",
  "messages": [
    {
      "target_group_index": 0,
      "target_name": "heavy-data-users",
      "message_drafts": [
        {
          "message_draft_index": 1,
          "message_text": "..."
        }
      ],
      "strategy_meta": {
        "mode": "reuse",
        "reason": "Past high-performing messages fit this segment pattern.",
        "references": ["promotion-101"]
      }
    }
  ]
}
```

## 요청 흐름

### 세그먼트 생성 흐름

1. 백엔드가 `/cluster-customers`를 호출합니다.
2. `app.py`가 `CampaignStrategyAgent.decide_clustering_strategy()`를 실행합니다.
3. Agent가 History Provider로 유사 프로모션을 검색합니다.
4. Agent가 전략 모드를 정합니다.
5. 해당 전략 정보를 포함한 프롬프트로 LLM이 세그먼트를 생성합니다.
6. 응답에 `clusters`와 `strategy_meta`를 함께 반환합니다.

### 메시지 생성 흐름

1. 백엔드가 `/generate-messages`를 호출합니다.
2. `create_message.py`가 세그먼트별로 `decide_message_strategy()`를 실행합니다.
3. Agent가 과거 메시지 전략을 참고할지 판단합니다.
4. 메시지 초안을 생성합니다.
5. 세그먼트별 결과에 `strategy_meta`를 포함해 반환합니다.

## 파일 구성

- `app.py`
  - Flask 엔드포인트 진입점
  - Strategy Agent 생성 및 API 연결
- `strategy_agent.py`
  - 전략 판단 Agent 본체
  - 세그먼트 전략 / 메시지 전략 결정
- `history_provider.py`
  - 과거 프로모션 조회 계층
  - MCP, 웹 플랫폼, Notion, mock provider 포함
- `create_message.py`
  - 세그먼트별 메시지 초안 생성
  - Agent 결과를 메시지 생성에 반영
- `upload_pdf.py`
  - PDF 텍스트 추출 및 프로모션 필드 파싱
- `requirements.txt`
  - Python 의존성 목록
- `Dockerfile`
  - 컨테이너 빌드 설정

## 환경 변수

### 필수

- `OPENAI_API_KEY`

### History Provider 관련

- `PROMOTION_HISTORY_SOURCE`
- `PROMOTION_HISTORY_WEB_BASE_URL`
- `PROMOTION_HISTORY_WEB_API_KEY`
- `PROMOTION_HISTORY_MCP_URL`
- `NOTION_API_KEY`
- `NOTION_DATABASE_ID`
- `NOTION_VERSION`
- `ENABLE_SAMPLE_HISTORY`

### 성과 가중치 관련

- `WEIGHT_CLICK_RATE`
- `WEIGHT_PARTICIPATION_RATE`
- `WEIGHT_CONVERSION_RATE`

## 로컬 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

최소 실행 기준:

```bash
OPENAI_API_KEY=your_key
```

예시:

```bash
PROMOTION_HISTORY_SOURCE=web_platform
PROMOTION_HISTORY_WEB_BASE_URL=https://marketing-platform-app.redriver-ce1c37ed.japaneast.azurecontainerapps.io
WEIGHT_CLICK_RATE=0.5
WEIGHT_PARTICIPATION_RATE=0.3
WEIGHT_CONVERSION_RATE=0.2
```

### 3. 서버 실행

```bash
python app.py
```

기본 포트는 `5000`입니다.

## 기술 스택

- Python 3.11
- Flask
- OpenAI API
- LangChain
- pdfplumber
- requests
- Docker
- Azure Container Apps

## 배포

- 모든 서버는 Azure Container Apps에 배포되어 있습니다.
- 이 저장소는 AI 기능 전용 서비스로 동작하며, 프론트엔드가 직접 호출하기보다 백엔드를 통해 사용됩니다.
- MCP를 통해 연결한 별도 플랫폼 주소는 아래와 같습니다.
  - https://marketing-platform-app.redriver-ce1c37ed.japaneast.azurecontainerapps.io/

## 관련 저장소

- Frontend: https://github.com/likemin35/Work_experience_program_FE
- Backend: https://github.com/likemin35/Work_experience_program_BE
