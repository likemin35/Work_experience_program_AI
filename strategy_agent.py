import json
import os
from typing import Any, Dict, List

from history_provider import build_history_provider


class CampaignStrategyAgent:
    """
    Agent-like orchestrator:
    1) fetches similar historical promotions via provider (MCP extension point)
    2) asks LLM to decide whether to reuse past strategy or create a new one
    """

    def __init__(self, client):
        self.client = client
        self.history_provider = build_history_provider()

    def decide_clustering_strategy(
        self,
        campaign: Dict[str, Any],
        customers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        customer_preview = "\n".join(
            f"- ({c.get('customerId')}) {c.get('description', '')[:180]}"
            for c in customers[:25]
        )
        history = self.history_provider.search_similar_promotions(
            campaign=campaign,
            query_text=f"{campaign.get('title', '')}\n{campaign.get('coreBenefitText', '')}",
            limit=5,
        )
        ranked = self._rank_by_performance(history)
        history_json = json.dumps([h.to_dict() for h in ranked], ensure_ascii=False)

        if not history:
            return {
                "strategy_mode": "new",
                "reason": "No historical promotions were available from MCP/history source.",
                "segmentation_guidance": "Create fresh groups from current campaign and customer data.",
                "history_references": [],
            }

        prompt = f"""
You are a CRM segmentation strategist.
Decide whether we should reuse previous segmentation logic or design a new one.

[Campaign]
title: {campaign.get("title")}
benefit: {campaign.get("coreBenefitText")}

[Customer preview]
{customer_preview}

[Historical promotions]
{history_json}

Return JSON only:
{{
  "strategy_mode": "reuse" | "hybrid" | "new",
  "reason": "short rationale",
  "segmentation_guidance": "practical instruction for clustering prompt",
  "history_references": ["promotion_id", "..."]
}}
"""
        return self._safe_json_completion(prompt, fallback={
            "strategy_mode": "hybrid",
            "reason": "Fallback strategy due to parsing failure.",
            "segmentation_guidance": "Use current customer features, but keep group styles close to top-performing historical campaigns.",
            "history_references": [h.promotion_id for h in ranked[:2]],
        })

    def decide_message_strategy(
        self,
        campaign: Dict[str, Any],
        segment_name: str,
        segment_features: str,
    ) -> Dict[str, Any]:
        history = self.history_provider.search_similar_promotions(
            campaign=campaign,
            query_text=f"{campaign.get('title', '')}\n{segment_name}\n{segment_features}",
            limit=5,
        )
        ranked = self._rank_by_performance(history)
        history_json = json.dumps([h.to_dict() for h in ranked], ensure_ascii=False)

        if not history:
            return {
                "message_mode": "new",
                "reason": "No historical message performance data was found.",
                "message_guidance": "Write a new message tailored to this segment.",
                "history_references": [],
            }

        prompt = f"""
You are a CRM copy strategist.
Decide if this segment should reuse message tone/pattern from historical promotions.

[Campaign]
title: {campaign.get("title")}
benefit: {campaign.get("coreBenefitText")}

[Segment]
name: {segment_name}
features: {segment_features}

[Historical promotions]
{history_json}

Return JSON only:
{{
  "message_mode": "reuse" | "hybrid" | "new",
  "reason": "short rationale",
  "message_guidance": "practical writing instruction",
  "history_references": ["promotion_id", "..."]
}}
"""
        return self._safe_json_completion(prompt, fallback={
            "message_mode": "hybrid",
            "reason": "Fallback strategy due to parsing failure.",
            "message_guidance": "Keep reliable wording from high-performing examples and adapt to current segment features.",
            "history_references": [h.promotion_id for h in ranked[:2]],
        })

    def _safe_json_completion(self, prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
            return json.loads(content)
        except Exception:
            return fallback

    def _rank_by_performance(self, history: List[Any]) -> List[Any]:
        return sorted(history, key=self._score_history, reverse=True)

    def _score_history(self, item: Any) -> float:
        ctr_weight = _to_float_env("WEIGHT_CLICK_RATE", 0.5)
        participation_weight = _to_float_env("WEIGHT_PARTICIPATION_RATE", 0.3)
        conversion_weight = _to_float_env("WEIGHT_CONVERSION_RATE", 0.2)

        ctr = float(item.click_through_rate or 0.0)
        participation = float(item.participation_rate or 0.0)
        conversion = float(item.conversion_rate or 0.0)

        return (ctr * ctr_weight) + (participation * participation_weight) + (conversion * conversion_weight)


def _to_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default
