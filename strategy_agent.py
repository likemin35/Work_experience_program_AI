import os
from typing import Any, Dict, List, Tuple

from history_provider import HistoricalPromotion, build_history_provider


class CampaignStrategyAgent:
    """
    Strategy agent:
    - fetch historical promotions from MCP/Web provider
    - compare actual metrics vs expected metrics
    - decide strategy mode: reuse / hybrid / new
    """

    def __init__(self, client=None):
        # client is kept for backward compatibility, but strategy is rule-based now.
        self.client = client
        self.history_provider = build_history_provider()

    def decide_clustering_strategy(
        self,
        campaign: Dict[str, Any],
        customers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        history = self.history_provider.search_similar_promotions(
            campaign=campaign,
            query_text=f"{campaign.get('title', '')}\n{campaign.get('coreBenefitText', '')}",
            limit=5,
        )
        ranked = self._rank_by_actual_performance(history)

        if not ranked:
            return {
                "strategy_mode": "new",
                "reason": "No historical promotions were available from MCP/history source.",
                "segmentation_guidance": "Create fresh groups from current campaign and customer data.",
                "history_references": [],
            }

        mode, reason, refs = self._decide_mode_with_reason(ranked)
        guidance = {
            "reuse": "Reuse historical grouping pattern and prioritize segments that previously outperformed expected KPI.",
            "hybrid": "Partially reuse high-performing segment rules, but adapt boundaries/features for current customers.",
            "new": "Design a new clustering strategy from current campaign + customer profile, avoiding underperforming historical patterns.",
        }[mode]

        return {
            "strategy_mode": mode,
            "reason": reason,
            "segmentation_guidance": guidance,
            "history_references": refs,
        }

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
        ranked = self._rank_by_actual_performance(history)

        if not ranked:
            return {
                "message_mode": "new",
                "reason": "No historical message performance data was found.",
                "message_guidance": "Write a new message tailored to this segment.",
                "history_references": [],
            }

        mode, reason, refs = self._decide_mode_with_reason(ranked)
        guidance = {
            "reuse": "Reuse tone/structure from historically over-performing messages and adapt product facts only.",
            "hybrid": "Keep proven opening/CTA structure, but rewrite segment-specific body for this campaign.",
            "new": "Create a new message concept because historical performance underperformed expected KPI.",
        }[mode]

        return {
            "message_mode": mode,
            "reason": reason,
            "message_guidance": guidance,
            "history_references": refs,
        }

    def _decide_mode_with_reason(
        self,
        ranked: List[HistoricalPromotion],
    ) -> Tuple[str, str, List[str]]:
        rows = []
        for item in ranked:
            actual, expected = self._scores(item)
            if expected is None:
                continue
            gap = actual - expected
            rows.append((item, actual, expected, gap))

        # If expected metrics are missing, keep partial reuse as safe default.
        if not rows:
            refs = [h.promotion_id for h in ranked[:2]]
            return (
                "hybrid",
                "Expected KPI values were missing in history, so partially reusing proven patterns.",
                refs,
            )

        avg_gap = sum(r[3] for r in rows) / len(rows)
        over_threshold = _to_float_env("KPI_OVER_GAP_THRESHOLD", 0.010)
        under_threshold = _to_float_env("KPI_UNDER_GAP_THRESHOLD", -0.010)
        neutral_band = _to_float_env("KPI_NEUTRAL_BAND", 0.006)

        over_refs = [r[0].promotion_id for r in rows if r[3] >= neutral_band][:3]
        under_refs = [r[0].promotion_id for r in rows if r[3] <= -neutral_band][:3]
        top_refs = [h.promotion_id for h in ranked[:3]]

        if avg_gap >= over_threshold:
            refs = over_refs or top_refs[:2]
            reason = (
                f"Historical campaigns exceeded expected KPI on average "
                f"(avg_gap={avg_gap:.4f}), so reuse strategy is recommended."
            )
            return "reuse", reason, refs

        if avg_gap <= under_threshold:
            refs = under_refs or top_refs[:2]
            reason = (
                f"Historical campaigns underperformed expected KPI on average "
                f"(avg_gap={avg_gap:.4f}), so a new strategy is recommended."
            )
            return "new", reason, refs

        refs = top_refs[:2]
        reason = (
            f"Historical campaigns were near expected KPI "
            f"(avg_gap={avg_gap:.4f}), so partially reusing strategy is recommended."
        )
        return "hybrid", reason, refs

    def _rank_by_actual_performance(self, history: List[HistoricalPromotion]) -> List[HistoricalPromotion]:
        return sorted(history, key=self._actual_score, reverse=True)

    def _actual_score(self, item: HistoricalPromotion) -> float:
        ctr_weight = _to_float_env("WEIGHT_CLICK_RATE", 0.5)
        participation_weight = _to_float_env("WEIGHT_PARTICIPATION_RATE", 0.3)
        conversion_weight = _to_float_env("WEIGHT_CONVERSION_RATE", 0.2)

        ctr = float(item.click_through_rate or 0.0)
        participation = float(item.participation_rate or 0.0)
        conversion = float(item.conversion_rate or 0.0)
        return (ctr * ctr_weight) + (participation * participation_weight) + (conversion * conversion_weight)

    def _expected_score(self, item: HistoricalPromotion) -> float:
        ctr_weight = _to_float_env("WEIGHT_CLICK_RATE", 0.5)
        participation_weight = _to_float_env("WEIGHT_PARTICIPATION_RATE", 0.3)
        conversion_weight = _to_float_env("WEIGHT_CONVERSION_RATE", 0.2)

        ctr = float(item.expected_click_through_rate or 0.0)
        participation = float(item.expected_participation_rate or 0.0)
        conversion = float(item.expected_conversion_rate or 0.0)
        return (ctr * ctr_weight) + (participation * participation_weight) + (conversion * conversion_weight)

    def _scores(self, item: HistoricalPromotion) -> Tuple[float, Any]:
        actual = self._actual_score(item)
        has_expected = (
            item.expected_participation_rate is not None
            or item.expected_conversion_rate is not None
            or item.expected_click_through_rate is not None
        )
        if not has_expected:
            return actual, None
        return actual, self._expected_score(item)


def _to_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default
