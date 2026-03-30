# Work_experience_program_AI

## Strategy Agent + History Source

AI server can now choose segmentation/message strategy (`reuse`, `hybrid`, `new`)
based on historical promotion performance.

### Environment variables

- `PROMOTION_HISTORY_SOURCE`: `mock` | `web_platform` | `mcp` | `notion_api` (default: `web_platform`)
- `PROMOTION_HISTORY_WEB_BASE_URL`: base URL for web platform (default: `https://mock-marketing-platform.local`)
- `PROMOTION_HISTORY_WEB_API_KEY`: API key for web platform (optional)
- If `PROMOTION_HISTORY_WEB_BASE_URL` contains `mock`, built-in mock records are returned.
- `PROMOTION_HISTORY_MCP_URL`: MCP gateway URL when source is `mcp`
- `NOTION_API_KEY`: Notion integration token when source is `notion_api`
- `NOTION_DATABASE_ID`: Notion DB id when source is `notion_api`
- `NOTION_VERSION`: Notion API version (default: `2022-06-28`)

Performance weighting (rule engine):
- `WEIGHT_CLICK_RATE` (default: `0.5`)
- `WEIGHT_PARTICIPATION_RATE` (default: `0.3`)
- `WEIGHT_CONVERSION_RATE` (default: `0.2`)
