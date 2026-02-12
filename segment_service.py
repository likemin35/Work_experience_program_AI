# segment_service.py
from rag_utils_target import query_chroma_targeting

def match_customer_to_segment(customer_text: str, top_k: int = 5):
    results = query_chroma_targeting(
        query_texts=[customer_text],
        n_results=top_k,
        where_filter=None
    )

    if not results:
        return {
            "segmentLabel": "기타 고객군",
            "score": 0.0,
            "document": None
        }

    best = results[0]
    return {
        "segmentLabel": best["metadata"].get("segment_label"),
        "score": 1 - best["distance"],
        "document": best["document"]
    }
