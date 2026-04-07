# Message AI Server

CRM 마케팅 메시지 생성을 위한 AI 서버입니다.

## Environment variables

- `PROMOTION_HISTORY_SOURCE`: `mock` | `web_platform` | `mcp` | `notion_api` (default: `web_platform`)
- `PROMOTION_HISTORY_WEB_BASE_URL`: web platform base URL
- `PROMOTION_HISTORY_WEB_API_KEY`: web platform API key (optional)
- `PROMOTION_HISTORY_MCP_URL`: MCP endpoint URL
- `NOTION_API_KEY`, `NOTION_DATABASE_ID`, `NOTION_VERSION`: Notion 연동용

Performance weighting:
- `WEIGHT_CLICK_RATE` (default: `0.5`)
- `WEIGHT_PARTICIPATION_RATE` (default: `0.3`)
- `WEIGHT_CONVERSION_RATE` (default: `0.2`)

KPI gap thresholds:
- `KPI_OVER_GAP_THRESHOLD` (default: `0.010`)
- `KPI_UNDER_GAP_THRESHOLD` (default: `-0.010`)
- `KPI_NEUTRAL_BAND` (default: `0.006`)

## 2026-04-07 Update

### Updated files
- `history_provider.py`
- `strategy_agent.py`

### What changed
- 히스토리 데이터 모델에 예상 KPI 필드 추가:
  - `expected_participation_rate`
  - `expected_conversion_rate`
  - `expected_click_through_rate`
- Agent 전략 판단을 예상 대비 실제 성과 기반 규칙으로 변경:
  - 평균 성과가 예상치보다 높으면 `reuse`
  - 평균 성과가 예상치보다 낮으면 `new`
  - 평균 성과가 유사하면 `hybrid`
- 판단 결과에 근거(`reason`)와 참조 프로모션(`history_references`)을 함께 반환하도록 강화
