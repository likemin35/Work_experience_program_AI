import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class HistoricalPromotion:
    promotion_id: str
    title: str
    summary: str
    target_description: str
    participation_rate: Optional[float] = None
    conversion_rate: Optional[float] = None
    click_through_rate: Optional[float] = None
    expected_participation_rate: Optional[float] = None
    expected_conversion_rate: Optional[float] = None
    expected_click_through_rate: Optional[float] = None
    message_example: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PromotionHistoryProvider:
    def search_similar_promotions(
        self,
        campaign: Dict[str, Any],
        query_text: str,
        limit: int = 5,
    ) -> List[HistoricalPromotion]:
        raise NotImplementedError


class MockPromotionHistoryProvider(PromotionHistoryProvider):
    """
    Safe default provider.
    Keeps runtime stable until MCP/Notion datasource is connected.
    """

    def search_similar_promotions(
        self,
        campaign: Dict[str, Any],
        query_text: str,
        limit: int = 5,
    ) -> List[HistoricalPromotion]:
        if os.getenv("ENABLE_SAMPLE_HISTORY", "false").lower() != "true":
            return []

        samples = [
            HistoricalPromotion(
                promotion_id="sample-1",
                title="Weekend Data Booster",
                summary="Extra data bundle offer for high mobile-data users.",
                target_description="Users with high data usage and video streaming behavior.",
                participation_rate=0.34,
                conversion_rate=0.14,
                click_through_rate=0.29,
                expected_participation_rate=0.30,
                expected_conversion_rate=0.12,
                expected_click_through_rate=0.26,
                message_example="For customers who stream often, we prepared extra weekend data.",
            ),
            HistoricalPromotion(
                promotion_id="sample-2",
                title="Dormant User Comeback",
                summary="Benefit for low-activity users to reactivate app/service visits.",
                target_description="Users with low recent app usage and low interaction counts.",
                participation_rate=0.18,
                conversion_rate=0.09,
                click_through_rate=0.17,
                expected_participation_rate=0.20,
                expected_conversion_rate=0.10,
                expected_click_through_rate=0.18,
                message_example="We prepared a simple comeback benefit just for returning customers.",
            ),
        ]
        return samples[:limit]


class MCPPromotionHistoryProvider(PromotionHistoryProvider):
    def __init__(self, endpoint: str, timeout_seconds: int = 8):
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def search_similar_promotions(
        self,
        campaign: Dict[str, Any],
        query_text: str,
        limit: int = 5,
    ) -> List[HistoricalPromotion]:
        try:
            response = requests.post(
                f"{self.endpoint}/search",
                json={
                    "campaign": campaign,
                    "query_text": query_text,
                    "limit": limit,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            items = payload.get("items", []) if isinstance(payload, dict) else payload
            return [_to_promotion(item) for item in items][:limit]
        except Exception:
            return []


class WebPlatformPromotionHistoryProvider(PromotionHistoryProvider):
    """
    Preferred production shape: web platform + DB.
    For now, supports a mock URL fallback for quick integration.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout_seconds: int = 8,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def search_similar_promotions(
        self,
        campaign: Dict[str, Any],
        query_text: str,
        limit: int = 5,
    ) -> List[HistoricalPromotion]:
        if "mock" in self.base_url:
            return self._mock_items(limit)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                f"{self.base_url}/api/promotions/search",
                headers=headers,
                json={
                    "campaign": campaign,
                    "query_text": query_text,
                    "limit": limit,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            items = payload.get("items", []) if isinstance(payload, dict) else payload
            return [_to_promotion(item) for item in items][:limit]
        except Exception:
            return []

    def _mock_items(self, limit: int) -> List[HistoricalPromotion]:
        data = [
            HistoricalPromotion(
                promotion_id="web-mock-101",
                title="Premium Data Pack Upgrade",
                summary="High data-usage segment upsell campaign.",
                target_description="Heavy data users, video/streaming focused.",
                participation_rate=0.37,
                conversion_rate=0.16,
                click_through_rate=0.33,
                expected_participation_rate=0.35,
                expected_conversion_rate=0.15,
                expected_click_through_rate=0.31,
                message_example="데이터 사용량이 높은 고객님께 업그레이드 혜택을 준비했습니다.",
            ),
            HistoricalPromotion(
                promotion_id="web-mock-102",
                title="Reactivation Benefit",
                summary="Re-engagement campaign for dormant customers.",
                target_description="Low activity users over last 60 days.",
                participation_rate=0.22,
                conversion_rate=0.11,
                click_through_rate=0.19,
                expected_participation_rate=0.25,
                expected_conversion_rate=0.12,
                expected_click_through_rate=0.21,
                message_example="최근 이용이 뜸했던 고객님께 복귀 혜택을 안내드립니다.",
            ),
            HistoricalPromotion(
                promotion_id="web-mock-103",
                title="Family Bundle Promotion",
                summary="Bundle promotion for multi-line households.",
                target_description="Households with family-plan potential.",
                participation_rate=0.29,
                conversion_rate=0.13,
                click_through_rate=0.24,
                expected_participation_rate=0.28,
                expected_conversion_rate=0.12,
                expected_click_through_rate=0.23,
                message_example="가족 결합 시 추가 혜택을 받을 수 있는 프로모션입니다.",
            ),
        ]
        return data[:limit]


class NotionPromotionHistoryProvider(PromotionHistoryProvider):
    def __init__(
        self,
        token: str,
        database_id: str,
        notion_version: str = "2022-06-28",
        timeout_seconds: int = 8,
    ):
        self.token = token
        self.database_id = database_id
        self.notion_version = notion_version
        self.timeout_seconds = timeout_seconds
        self.base_url = "https://api.notion.com/v1"

    def search_similar_promotions(
        self,
        campaign: Dict[str, Any],
        query_text: str,
        limit: int = 5,
    ) -> List[HistoricalPromotion]:
        pages = self._query_database(page_size=max(15, limit * 3))
        if not pages:
            return []

        title_hint = f"{campaign.get('title', '')} {campaign.get('coreBenefitText', '')}"
        scored = []
        for page in pages:
            promotion = self._to_history(page)
            if promotion is None:
                continue
            sim = _keyword_similarity(f"{promotion.title} {promotion.summary}", f"{title_hint} {query_text}")
            scored.append((sim, promotion))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    def _query_database(self, page_size: int) -> List[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Notion-Version": self.notion_version,
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                f"{self.base_url}/databases/{self.database_id}/query",
                headers=headers,
                json={"page_size": page_size},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception:
            return []

    def _to_history(self, page: Dict[str, Any]) -> Optional[HistoricalPromotion]:
        props = page.get("properties", {})
        promotion_id = str(page.get("id", ""))
        title = _extract_title(props, _env("NOTION_PROP_TITLE", "title"))
        if not title:
            title = _extract_title_any(props)

        summary = _extract_rich_text(props, _env("NOTION_PROP_SUMMARY", "summary"))
        target_description = _extract_rich_text(props, _env("NOTION_PROP_TARGET", "target_description"))
        message_example = _extract_rich_text(props, _env("NOTION_PROP_MESSAGE", "message_example"))

        return HistoricalPromotion(
            promotion_id=promotion_id,
            title=title or "Untitled Promotion",
            summary=summary,
            target_description=target_description,
            participation_rate=_extract_number(props, _env("NOTION_PROP_PARTICIPATION_RATE", "participation_rate")),
            conversion_rate=_extract_number(props, _env("NOTION_PROP_CONVERSION_RATE", "conversion_rate")),
            click_through_rate=_extract_number(props, _env("NOTION_PROP_CLICK_RATE", "click_through_rate")),
            expected_participation_rate=_extract_number(
                props, _env("NOTION_PROP_EXPECTED_PARTICIPATION_RATE", "expected_participation_rate")
            ),
            expected_conversion_rate=_extract_number(
                props, _env("NOTION_PROP_EXPECTED_CONVERSION_RATE", "expected_conversion_rate")
            ),
            expected_click_through_rate=_extract_number(
                props, _env("NOTION_PROP_EXPECTED_CLICK_RATE", "expected_click_through_rate")
            ),
            message_example=message_example,
        )


def build_history_provider() -> PromotionHistoryProvider:
    """
    Extension point:
    - Today: safe mock provider.
    - Later: replace with MCP-backed Notion or internal performance DB provider.
    """
    source = os.getenv("PROMOTION_HISTORY_SOURCE", "web_platform").lower()

    if source == "web_platform":
        base_url = os.getenv(
            "PROMOTION_HISTORY_WEB_BASE_URL",
            "https://mock-marketing-platform.local",
        ).strip()
        api_key = os.getenv("PROMOTION_HISTORY_WEB_API_KEY", "").strip() or None
        if base_url:
            return WebPlatformPromotionHistoryProvider(
                base_url=base_url,
                api_key=api_key,
            )

    if source == "mcp":
        endpoint = os.getenv("PROMOTION_HISTORY_MCP_URL", "").strip()
        if endpoint:
            return MCPPromotionHistoryProvider(endpoint=endpoint)

    if source == "notion_api":
        token = os.getenv("NOTION_API_KEY", "").strip()
        db_id = os.getenv("NOTION_DATABASE_ID", "").strip()
        if token and db_id:
            return NotionPromotionHistoryProvider(
                token=token,
                database_id=db_id,
                notion_version=os.getenv("NOTION_VERSION", "2022-06-28"),
            )

    return MockPromotionHistoryProvider()


def _to_promotion(item: Dict[str, Any]) -> HistoricalPromotion:
    return HistoricalPromotion(
        promotion_id=str(item.get("promotion_id", "")),
        title=str(item.get("title", "")),
        summary=str(item.get("summary", "")),
        target_description=str(item.get("target_description", "")),
        participation_rate=_to_float(item.get("participation_rate")),
        conversion_rate=_to_float(item.get("conversion_rate")),
        click_through_rate=_to_float(item.get("click_through_rate")),
        expected_participation_rate=_to_float(item.get("expected_participation_rate")),
        expected_conversion_rate=_to_float(item.get("expected_conversion_rate")),
        expected_click_through_rate=_to_float(item.get("expected_click_through_rate")),
        message_example=item.get("message_example"),
    )


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _extract_title(props: Dict[str, Any], key: str) -> str:
    title_obj = props.get(key, {})
    arr = title_obj.get("title", [])
    return "".join(chunk.get("plain_text", "") for chunk in arr).strip()


def _extract_title_any(props: Dict[str, Any]) -> str:
    for val in props.values():
        if isinstance(val, dict) and "title" in val:
            arr = val.get("title", [])
            text = "".join(chunk.get("plain_text", "") for chunk in arr).strip()
            if text:
                return text
    return ""


def _extract_rich_text(props: Dict[str, Any], key: str) -> str:
    obj = props.get(key, {})
    arr = obj.get("rich_text", [])
    return "".join(chunk.get("plain_text", "") for chunk in arr).strip()


def _extract_number(props: Dict[str, Any], key: str) -> Optional[float]:
    obj = props.get(key, {})
    if "number" in obj:
        return _to_float(obj.get("number"))
    return None


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _keyword_similarity(text_a: str, text_b: str) -> float:
    a_tokens = {t for t in _normalize(text_a).split() if t}
    b_tokens = {t for t in _normalize(text_b).split() if t}
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens.intersection(b_tokens))
    return overlap / max(len(a_tokens), 1)


def _normalize(text: str) -> str:
    return (
        (text or "")
        .lower()
        .replace("/", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace("\n", " ")
    )
