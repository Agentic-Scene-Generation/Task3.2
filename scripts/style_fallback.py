"""Explicit unknown-style label, independent of confidence and the 17-class ontology."""

import copy


def with_style_fallback(annotation):
    value = copy.deepcopy(annotation or {})
    labels = sorted(
        {s["level_1_id"] for s in value.get("styles", []) if s.get("level_1_id")}
    )
    # Preserve legacy explicitly supported styles without making unspecified a style.
    if value.get("schema_version") == "asset_style@1.0":
        labels = sorted(
            {
                s["style_id"]
                for s in value.get("intrinsic_styles", [])
                if s.get("style_id") not in (None, "unspecified", "unknown_style")
            }
        )
    fallback = not labels
    value["style_label_ids"] = labels or ["unknown_style"]
    value["style_is_fallback"] = fallback
    value["style_retrieval_priority_tier"] = 1 if fallback else 0
    value["style_fallback"] = (
        {
            "label_id": "unknown_style",
            "label_zh": "风格未识别",
            "reason": (
                "max_judgments_exhausted"
                if value.get("judgment_count", 0) >= value.get("max_judgments", 5)
                else "inconclusive_or_pending"
            ),
            "is_semantic_style": False,
            "policy": "unknown_style_low_priority@1.0",
        }
        if fallback
        else None
    )
    return value


def rerank_style_candidates(candidates, overlay, top_k):
    """Stable rerank within the retrieved pool. Original scores remain unchanged."""
    enriched = []
    for item in candidates:
        row = dict(item)
        uid = str(row.get("asset_id") or row.get("asset_uid") or "")
        annotation = overlay.get(uid, overlay.get(uid.removeprefix("hssd:")))
        if annotation is None:
            # This overlay covers non-generated assets. Do not falsely relabel an
            # out-of-scope generated asset whose style lives in another registry.
            row.update(
                style_retrieval_priority_tier=0, style_priority_policy_applied=False
            )
            enriched.append(row)
            continue
        info = with_style_fallback(annotation)
        row.update(
            {
                k: info[k]
                for k in (
                    "style_label_ids",
                    "style_is_fallback",
                    "style_retrieval_priority_tier",
                )
            }
        )
        row["style_priority_policy_applied"] = True
        enriched.append(row)
    return sorted(enriched, key=lambda r: r["style_retrieval_priority_tier"])[:top_k]
