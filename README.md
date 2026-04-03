# Message AI Server

메시지 생성 프로젝트의 AI 서버 저장소입니다. 프로모션 PDF에서 핵심 정보를 추출하고, 고객 CSV를 바탕으로 세그먼트를 나누며, 세그먼트별 맞춤형 마케팅 메시지 초안을 생성합니다.

## 프로젝트 개요

- 이 저장소는 3개 레포로 분리된 전체 시스템 중 AI 서버를 담당합니다.
- Azure Container Apps 환경에 배포되어 백엔드 서버와 연동됩니다.
- 프론트 서비스 주소: https://message-fe-app.redriver-ce1c37ed.japaneast.azurecontainerapps.io/
- 별도 연동 플랫폼 주소: https://marketing-platform-app.redriver-ce1c37ed.japaneast.azurecontainerapps.io/

## 주요 기능

- 프로모션 PDF 텍스트 추출
- OpenAI 기반 프로모션 정보 구조화
- 고객 설명 기반 세그먼트 클러스터링
- 세그먼트별 개인화 메시지 초안 생성
- Flask API 형태로 백엔드 서버에 AI 기능 제공

## Strategy Agent + History Source

이 저장소의 README에는 최근 이력 기반 전략 선택 관련 설정도 반영되어 있었습니다. 현재 문서에도 그 내용을 함께 정리합니다.

- 세그먼트 전략 또는 메시지 전략을 `reuse`, `hybrid`, `new` 방식으로 확장할 수 있는 구조를 염두에 두고 있습니다.
- 운영 환경에서 프로모션 이력 데이터를 외부 플랫폼, MCP, Notion 등의 소스로 연결하는 설정을 둘 수 있습니다.

### 관련 환경 변수

- `PROMOTION_HISTORY_SOURCE`: `mock`, `web_platform`, `mcp`, `notion_api`
- `PROMOTION_HISTORY_WEB_BASE_URL`: 외부 플랫폼 기본 주소
- `PROMOTION_HISTORY_WEB_API_KEY`: 외부 플랫폼 API 키
- `PROMOTION_HISTORY_MCP_URL`: MCP 게이트웨이 주소
- `NOTION_API_KEY`: Notion 연동 토큰
- `NOTION_DATABASE_ID`: Notion 데이터베이스 ID
- `NOTION_VERSION`: Notion API 버전
- `WEIGHT_CLICK_RATE`: 성과 가중치
- `WEIGHT_PARTICIPATION_RATE`: 참여율 가중치
- `WEIGHT_CONVERSION_RATE`: 전환율 가중치

프로젝트 운영 방식상 MCP 기반 외부 플랫폼 연동 주소는 아래와 같습니다.

- https://marketing-platform-app.redriver-ce1c37ed.japaneast.azurecontainerapps.io/

## API 역할

### `POST /ai/campaign/extract`

- 업로드된 PDF를 읽고 프로모션 제목, 핵심 혜택, 연결 URL을 추출합니다.

### `POST /cluster-customers`

- 캠페인 정보와 고객 설명 목록을 받아 타겟 그룹을 생성합니다.

### `POST /generate-messages`

- 세그먼트별 특징을 입력받아 메시지 초안을 생성합니다.

### `GET /`

- 헬스 체크용 엔드포인트입니다.

## 기술 스택

- Python 3.11
- Flask
- OpenAI API
- LangChain
- pdfplumber
- Docker
- Azure Container Apps

## 파일 구성

- `app.py`: Flask 엔드포인트 진입점
- `upload_pdf.py`: PDF 텍스트 추출 및 프로모션 정보 파싱
- `create_message.py`: 세그먼트별 메시지 생성 로직
- `segment_service.py`, `main.py`: 실험 또는 보조 로직
- `requirements.txt`: Python 의존성 목록
- `Dockerfile`: 컨테이너 이미지 빌드 설정

## 로컬 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

필수 환경 변수:

- `OPENAI_API_KEY`

### 3. 서버 실행

```bash
python app.py
```

기본 포트는 `5000`입니다.

## 백엔드 연동 방식

- 백엔드는 PDF 업로드 시 `/ai/campaign/extract`를 호출합니다.
- 고객 CSV 업로드 후 세그먼트 생성 시 `/cluster-customers`를 호출합니다.
- 메시지 생성 단계에서 `/generate-messages`를 호출합니다.
- Azure Container Apps 내부 통신 기준으로 백엔드가 이 AI 서버를 호출하도록 구성되어 있습니다.

## 배포

- 모든 서버는 Azure Container Apps에 배포되어 있습니다.
- 이 저장소는 AI 기능 전용 서비스로 동작하며, 프론트에서 직접 호출하기보다 백엔드를 통해 사용됩니다.
- 운영 환경에서는 OpenAI API 키를 환경 변수로 주입해야 합니다.

## 관련 저장소

- Frontend: https://github.com/likemin35/Work_experience_program_FE
- Backend: https://github.com/likemin35/Work_experience_program_BE
