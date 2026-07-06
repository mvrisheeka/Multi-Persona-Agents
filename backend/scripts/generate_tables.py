"""
generate_tables.py
------------------
Run AFTER routing_test_cli.py has saved a report Excel.
Produces two tables for the paper:

  TABLE II  — Per-class Precision / Recall / F1 / Support
  TABLE III — Routing cascade layer contribution

Usage:
    python generate_tables.py reports/routing_report_YYYYMMDD_HHMMSS.xlsx

Or drop generate_tables() call at the end of evaluate() in routing_test_cli.py.
"""

import sys
import json
import pandas as pd
from collections import defaultdict
import router as router_module

router_module.ROUTER_EVAL_MODE = True


def get_router_column(df: pd.DataFrame) -> str:
    if "router_prior_personas" in df.columns:
        return "router_prior_personas"
    return "predicted_personas"


# ─────────────────────────────────────────────────────────────────
#  HELPER: detect which routing tier handled a query
#  (mirrors the logic in router.py so you can tag each row)
# ─────────────────────────────────────────────────────────────────
def detect_tier(query: str, predicted_personas: list) -> str:
    emotional = router_module.detect_emotional_intent(query)
    decision = router_module.detect_decision_intent(query)

    if emotional and decision:
        return "Tier 1: Emotional Rule Classifier"
    if emotional:
        return "Tier 1: Emotional Rule Classifier"

    result = router_module.rule_router(query)
    if result:
        return "Tier 2: Full Rule-Based Router"

    result = router_module.trained_router(query)
    if result:
        return "Tier 3: Trained ML Router"

    semantic_result = router_module.semantic_router.predict(query)
    if semantic_result is not None:
        return "Tier 4: Semantic Similarity Fallback"

    return "Final Fallback: LLM / Default"

# ─────────────────────────────────────────────────────────────────
#  TABLE II — Per-class Precision / Recall / F1 / Support
# ─────────────────────────────────────────────────────────────────
def table_ii(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-label per-class metrics.
    Each row in df has:
        expected_personas   : "teacher, counselor"
        predicted_personas  : "counselor"
    """
    personas = ["teacher", "parent", "senior", "friend", "counselor"]
    router_col = get_router_column(df)
    rows = []

    for p in personas:
        tp = fp = fn = 0
        for _, row in df.iterrows():
            expected  = {x.strip() for x in str(row["expected_personas"]).lower().split(",")}
            predicted = {x.strip() for x in str(row[router_col]).lower().split(",")}

            if p in predicted and p in expected:
                tp += 1
            elif p in predicted and p not in expected:
                fp += 1
            elif p not in predicted and p in expected:
                fn += 1

        precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0.0
        recall    = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0.0
        f1        = round(2 * precision * recall / (precision + recall), 1) if (precision + recall) > 0 else 0.0
        support   = tp + fn  # total ground-truth positives for this class

        rows.append({
            "Persona":    p.capitalize(),
            "Precision (%)": precision,
            "Recall (%)":    recall,
            "F1-Score (%)":  f1,
            "Support":       support
        })

    # Micro-average
    total_tp = total_fp = total_fn = 0
    for p in personas:
        for _, row in df.iterrows():
            expected  = {x.strip() for x in str(row["expected_personas"]).lower().split(",")}
            predicted = {x.strip() for x in str(row[router_col]).lower().split(",")}
            if p in predicted and p in expected:     total_tp += 1
            elif p in predicted and p not in expected: total_fp += 1
            elif p not in predicted and p in expected: total_fn += 1

    mp = round(total_tp / (total_tp + total_fp) * 100, 1) if (total_tp+total_fp)>0 else 0
    mr = round(total_tp / (total_tp + total_fn) * 100, 1) if (total_tp+total_fn)>0 else 0
    mf = round(2*mp*mr/(mp+mr), 1) if (mp+mr)>0 else 0

    rows.append({
        "Persona": "Micro-Average",
        "Precision (%)": mp,
        "Recall (%)": mr,
        "F1-Score (%)": mf,
        "Support": total_tp + total_fn
    })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
#  TABLE III — Routing Cascade Layer Contribution
# ─────────────────────────────────────────────────────────────────
def table_iii(df: pd.DataFrame) -> pd.DataFrame:
    tier_counts = defaultdict(int)
    tier_correct = defaultdict(int)
    total = len(df)
    router_col = get_router_column(df)

    for _, row in df.iterrows():
        predicted = [x.strip() for x in str(row[router_col]).lower().split(",")]
        tier = detect_tier(row["query"], predicted)
        tier_counts[tier] += 1
        if row["correct"]:
            tier_correct[tier] += 1

    order = [
        "Tier 1: Emotional Rule Classifier",
        "Tier 2: Full Rule-Based Router",
        "Tier 3: Trained ML Router",
        "Tier 4: Semantic Similarity Fallback",
        "Final Fallback: LLM / Default"
    ]

    rows = []
    for tier in order:
        count   = tier_counts.get(tier, 0)
        correct = tier_correct.get(tier, 0)
        pct     = round(count / total * 100, 1) if total > 0 else 0
        acc     = round(correct / count * 100, 1) if count > 0 else 0
        rows.append({
            "Routing Tier":       tier,
            "Queries Resolved":   count,
            "% of Total":         f"{pct}%",
            "Tier Accuracy (%)":  acc
        })

    # totals row
    rows.append({
        "Routing Tier":      "TOTAL",
        "Queries Resolved":  total,
        "% of Total":        "100%",
        "Tier Accuracy (%)": round(df["correct"].sum() / total * 100, 1)
    })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
#  MAIN — call with path to existing report Excel
# ─────────────────────────────────────────────────────────────────
def generate_tables(report_excel_path: str):
    df = pd.read_excel(report_excel_path)
    df.columns = [c.strip().lower() for c in df.columns]
    router_col = get_router_column(df)

    t2 = table_ii(df)
    t3 = table_iii(df)

    print("\n" + "="*60)
    print("TABLE II — Per-Class Router-Prior Metrics")
    print("="*60)
    print(t2.to_string(index=False))

    print("\n" + "="*60)
    print("TABLE III — Routing Cascade Layer Contribution")
    print("="*60)
    print(t3.to_string(index=False))
    print(f"\nRouter prior column used: {router_col}")
    if "execution_personas" in df.columns:
        print("Execution personas are recorded separately and are not used in these routing tables.")

    # Save alongside the input report
    base = report_excel_path.replace(".xlsx", "")
    t2.to_excel(f"{base}_table2_perclass.xlsx", index=False)
    t3.to_excel(f"{base}_table3_cascade.xlsx",  index=False)

    print(f"\nSaved:")
    print(f"  {base}_table2_perclass.xlsx")
    print(f"  {base}_table3_cascade.xlsx")

    return t2, t3


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_tables.py reports/routing_report_XXXX.xlsx")
        sys.exit(1)
    generate_tables(sys.argv[1])
