# Message AI Server

메시지 생성 프로젝트의 AI 서버 저장소입니다. 이 서버는 프로모션 PDF에서 핵심 정보를 추출하고, 고객 세그먼트를 생성하며, 세그먼트별 맞춤형 CRM 메시지 초안을 생성합니다.

현재는 Spring Boot 백엔드와 Azure Container Apps Job worker가 이 FastAPI 서버를 호출해 비동기 캠페인 처리 흐름을 구성합니다.

## 서비스 개요

- 역할: AI 전용 FastAPI 서버
- 배포 환경: Azure Container Apps
- 프론트 서비스 주소: https://message-fe-app.redriver-ce1c37ed.japaneast.azurecontainerapps.io/
- MCP 연동 플랫폼 주소: https://marketing-platform-app.redriver-ce1c37ed.japaneast.azurecontainerapps.io/

## 주요 기능

- 프로모션 PDF 텍스트 추출
- OpenAI 기반 프로모션 정보 구조화
- 고객 설명 기반 세그먼트 클러스터링
- 세그먼트별 메시지 초안 생성
- 과거 프로모션 이력 기반 전략 선택
- MCP / 외부 플랫폼 / Notion 기반 이력 데이터 소스 확장

## 주요 API

### `POST /ai/campaign/extract`

- 프로모션 PDF를 입력받아 제목, 핵심 혜택, 상세 URL을 추출합니다.

### `POST /cluster-customers`

- 캠페인 정보와 고객 설명 목록을 받아 세그먼트를 생성합니다.
- 응답에는 `clusters`와 `strategy_meta`가 포함됩니다.

### `POST /generate-messages`

- 세그먼트별 특징을 입력받아 CRM 메시지 초안을 생성합니다.
- 응답에는 메시지 결과와 `strategy_meta`가 포함됩니다.

### `GET /`

- 헬스 체크 엔드포인트입니다.

## Strategy Agent

이 저장소의 Agent 기능은 과거 프로모션 이력과 성과 데이터를 바탕으로 세그먼트 전략 또는 메시지 전략을 고르는 오케스트레이션 레이어입니다.

### 전략 모드

- `reuse`
- `hybrid`
- `new`

### 주요 파일

- `app.py`
  - FastAPI 엔드포인트
- `strategy_agent.py`
  - 전략 선택 로직
- `history_provider.py`
  - MCP / 웹 플랫폼 / Notion 기반 이력 조회
- `upload_pdf.py`
  - PDF 추출 및 프로모션 필드 파싱
- `create_message.py`
  - 세그먼트별 메시지 생성

## 현재 백엔드 연동 방식

1. 백엔드 웹 앱이 업로드 시 `/ai/campaign/extract`를 호출합니다.
2. segmentation worker가 `/cluster-customers`를 호출합니다.
3. message generation worker가 `/generate-messages`를 호출합니다.
4. AI 서버는 결과를 즉시 반환하고, 상태 관리와 파일 저장은 백엔드가 담당합니다.

## 기술 스택

- Python 3.11
- FastAPI
- OpenAI API
- pdfplumber
- requests
- uvicorn
- Docker
- Azure Container Apps

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

### KPI / 성과 기반 판단

- `WEIGHT_CLICK_RATE`
- `WEIGHT_PARTICIPATION_RATE`
- `WEIGHT_CONVERSION_RATE`
- `KPI_OVER_GAP_THRESHOLD`
- `KPI_UNDER_GAP_THRESHOLD`
- `KPI_NEUTRAL_BAND`

## 로컬 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
OPENAI_API_KEY=your_key
```

### 3. 서버 실행

```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```

## 배포

- Azure Container Apps에 배포되어 있습니다.
- 비동기 Queue/Job 구조에서도 AI 서버 자체는 HTTP API 서버로 유지됩니다.
- worker는 이 저장소의 API를 호출해 세그먼트 생성과 메시지 생성을 수행합니다.

## 관련 저장소

- Backend: https://github.com/likemin35/Work_experience_program_BE
- Frontend: https://github.com/likemin35/Work_experience_program_FE
