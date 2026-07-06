"""
routing_analytics.py
--------------------
Drop this file next to routing_test_cli.py and run:
    python routing_analytics.py

It reads your existing routing report Excel (or re-runs the eval)
and prints all tables needed for the paper:

  Table II  — Overall routing performance (EMA, Top-1, P, R, F1)
# Continued
routing_analytics.py
--------------------
Drop this file next to routing_test_cli.py and run:
    python routing_analytics.py

It reads your existing routing report Excel (or re-runs the eval)
and prints all tables needed for the paper:

  Table II  — Overall routing performance (EMA, Top-1, P, R, F1)
  Table III — Cascade layer contribution breakdown
  Table IV  — Per-persona precision / recall / F1 / support
  Table V   — Single-persona Teacher baseline vs full system
  Bonus     — Top-20 misclassified queries for manual review
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
import json
import glob
from sklearn.metrics import precision_score, recall_score, f1_score

# ─── STEP 1: find the latest report or re-run eval ──────────────────────────

REPORT_DIR = "reports"

def load_latest_report():
    files = sorted(glob.glob(os.path.join(REPORT_DIR, "routing_report_*.xlsx")))
    if not files:
        print("No report found. Running evaluation first...")
        from routing_test_cli import evaluate
        evaluate("data/GroundTruth_test.xlsx")
        files = sorted(glob.glob(os.path.join(REPORT_DIR, "routing_report_*.xlsx")))
    latest = files[-1]
    print(f"Using report: {latest}\n")
    return pd.read_excel(latest)


df = load_latest_report()

# ─── normalise columns ───────────────────────────────────────────────────────

def parse_personas(cell):
    """'teacher, senior' → ['teacher', 'senior']"""
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [p.strip().lower() for p in str(cell).split(",")]


def get_router_column(df):
    if "router_prior_personas" in df.columns:
        return "router_prior_personas"
    return "predicted_personas"

ROUTER_COL = get_router_column(df)
df["expected_list"] = df["expected_personas"].apply(parse_personas)
df["predicted_list"] = df[ROUTER_COL].apply(parse_personas)

PERSONAS = ["teacher", "parent", "senior", "friend", "counselor"]
total = len(df)

# ════════════════════════════════════════════════════════════════════════════
#  TABLE II — Overall routing performance
# ════════════════════════════════════════════════════════════════════════════

# Exact Match Accuracy — predicted set == expected set exactly
ema = (df["expected_list"].apply(frozenset) == df["predicted_list"].apply(frozenset)).sum()
ema_pct = round(ema / total * 100, 2)

# Top-1 Accuracy — at least one predicted persona in expected set (what CLI already computes)
top1 = df["correct"].sum()
top1_pct = round(top1 / total * 100, 2)

# Multi-label Precision / Recall / F1
# Build binary matrices
def to_binary(lists, personas):
    mat = np.zeros((len(lists), len(personas)), dtype=int)
    for i, lst in enumerate(lists):
        for j, p in enumerate(personas):
            if p in lst:
                mat[i][j] = 1
    return mat

y_true = to_binary(df["expected_list"],  PERSONAS)
y_pred = to_binary(df["predicted_list"], PERSONAS)

micro_p  = round(precision_score(y_true, y_pred, average="micro", zero_division=0) * 100, 2)
micro_r  = round(recall_score(   y_true, y_pred, average="micro", zero_division=0) * 100, 2)
micro_f1 = round(f1_score(       y_true, y_pred, average="micro", zero_division=0) * 100, 2)

print("=" * 55)
print("TABLE II — Overall Router-Prior Performance")
print("=" * 55)
print(f"  Exact Match Accuracy (EMA) : {ema_pct:>6.2f} %   ({ema}/{total})")
print(f"  Top-1 Accuracy             : {top1_pct:>6.2f} %   ({top1}/{total})")
print(f"  Precision (micro)          : {micro_p:>6.2f} %")
print(f"  Recall    (micro)          : {micro_r:>6.2f} %")
print(f"  F1-Score  (micro)          : {micro_f1:>6.2f} %")
print(f"  Router Prior Column        : {ROUTER_COL}")
if "execution_personas" in df.columns:
    print("  Note                       : execution personas are recorded separately")
print()

# ════════════════════════════════════════════════════════════════════════════
#  TABLE III — Cascade layer contribution
#  We need to know WHICH tier resolved each query.
#  Re-run classify_intent with instrumentation.
# ════════════════════════════════════════════════════════════════════════════

print("Computing cascade layer breakdown (re-running router with instrumentation)...")

import router as router_module
router_module.ROUTER_EVAL_MODE = True

# Monkey-patch to track which tier fired
from router import (
    detect_emotional_intent, detect_decision_intent,
    rule_router, trained_router
)
from semantic_router import SemanticRouter

tier_counts = {"tier1_emotional": 0, "tier2_rule": 0,
               "tier3_ml": 0, "tier4_semantic": 0, "tier5_llm_fallback": 0}

def classify_with_tier(query):
    emotional = detect_emotional_intent(query)
    decision  = detect_decision_intent(query)

    if emotional and decision:
        tier_counts["tier1_emotional"] += 1
        return "tier1_emotional"
    if emotional:
        tier_counts["tier1_emotional"] += 1
        return "tier1_emotional"

    # Tier 2: rule router
    result = rule_router(query)
    if result:
        tier_counts["tier2_rule"] += 1
        return "tier2_rule"

    # Tier 3: ML router (if available)
    ml_result = trained_router(query)
    if ml_result:
        tier_counts["tier3_ml"] += 1
        return "tier3_ml"

    # Tier 4: semantic router
    sem = router_module.semantic_router.predict(query)
    if sem is not None:
        tier_counts["tier4_semantic"] += 1
        return "tier4_semantic"

    # Tier 5: LLM fallback (eval mode → friend)
    tier_counts["tier5_llm_fallback"] += 1
    return "tier5_llm_fallback"

df["tier"] = df["expected_list"].apply(lambda _: None)  # placeholder

for idx, row in df.iterrows():
    df.at[idx, "tier"] = classify_with_tier(row["query"])

print()
print("=" * 65)
print("TABLE III — Routing Cascade Layer Contribution")
print("=" * 65)
tier_labels = {
    "tier1_emotional":  "Tier 1: Emotional Rule Classifier",
    "tier2_rule":       "Tier 2: Full Rule-Based Router",
    "tier3_ml":         "Tier 3: Trained ML Router",
    "tier4_semantic":   "Tier 4: Semantic Similarity Fallback",
    "tier5_llm_fallback": "Final Fallback: LLM / Default",
}
for key, label in tier_labels.items():
    count = tier_counts[key]
    pct   = round(count / total * 100, 1)
    print(f"  {label:<42} {count:>4}  ({pct:.1f}%)")
print(f"  {'TOTAL':<42} {total:>4}  (100%)")
print()

# ════════════════════════════════════════════════════════════════════════════
#  TABLE IV — Per-persona precision / recall / F1 / support
# ════════════════════════════════════════════════════════════════════════════

p_per  = precision_score(y_true, y_pred, average=None, zero_division=0)
r_per  = recall_score(   y_true, y_pred, average=None, zero_division=0)
f1_per = f1_score(       y_true, y_pred, average=None, zero_division=0)
support = y_true.sum(axis=0)

print("=" * 70)
print("TABLE IV — Per-Persona Classification Performance")
print("=" * 70)
print(f"  {'Persona':<12}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
for i, persona in enumerate(PERSONAS):
    print(f"  {persona.capitalize():<12}  {p_per[i]*100:>9.2f}%  {r_per[i]*100:>7.2f}%  {f1_per[i]*100:>7.2f}%  {int(support[i]):>8}")
print()

# ════════════════════════════════════════════════════════════════════════════
#  TABLE V — Single-persona Teacher baseline vs full system
# ════════════════════════════════════════════════════════════════════════════

print("=" * 55)
print("TABLE V — Router Prior vs Single-Persona Baselines")
print("=" * 55)

for baseline_persona in ["teacher", "friend"]:
    baseline_pred = [[baseline_persona]] * total
    b_true = to_binary(df["expected_list"],  PERSONAS)
    b_pred = to_binary(baseline_pred,        PERSONAS)

    b_top1 = sum(
        1 for exp, pred in zip(df["expected_list"], baseline_pred)
        if set(pred) & set(exp)
    )
    b_top1_pct = round(b_top1 / total * 100, 2)
    b_p  = round(precision_score(b_true, b_pred, average="micro", zero_division=0) * 100, 2)
    b_r  = round(recall_score(   b_true, b_pred, average="micro", zero_division=0) * 100, 2)
    b_f1 = round(f1_score(       b_true, b_pred, average="micro", zero_division=0) * 100, 2)

    print(f"\n  Baseline: always route to '{baseline_persona.upper()}'")
    print(f"    Top-1 Accuracy : {b_top1_pct:.2f}%")
    print(f"    Precision      : {b_p:.2f}%")
    print(f"    Recall         : {b_r:.2f}%")
    print(f"    F1-Score       : {b_f1:.2f}%")

print(f"\n  Router Prior (used to guide full multi-persona execution)")
print(f"    Top-1 Accuracy : {top1_pct:.2f}%")
print(f"    Precision      : {micro_p:.2f}%")
print(f"    Recall         : {micro_r:.2f}%")
print(f"    F1-Score       : {micro_f1:.2f}%")
print()

# ════════════════════════════════════════════════════════════════════════════
#  BONUS — Top-20 misclassified queries for manual review
# ════════════════════════════════════════════════════════════════════════════

wrong = df[df["correct"] == False][["query", "expected_personas", ROUTER_COL, "tier"]].head(20)

print("=" * 80)
print("BONUS — First 20 Misclassified Queries (for manual review / dataset fixes)")
print("=" * 80)
if len(wrong) == 0:
    print("  No misclassifications found!")
else:
    for _, row in wrong.iterrows():
        print(f"  Query     : {row['query'][:70]}")
        print(f"  Expected  : {row['expected_personas']}")
        print(f"  Predicted : {row[ROUTER_COL]}")
        print(f"  Tier      : {row['tier']}")
        print()

# ════════════════════════════════════════════════════════════════════════════
#  SAVE all results to Excel (one sheet per table)
# ════════════════════════════════════════════════════════════════════════════

from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(REPORT_DIR, f"paper_tables_{ts}.xlsx")

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

    # Table II
    t2 = pd.DataFrame({
        "Metric": ["Exact Match Accuracy (EMA)", "Top-1 Accuracy",
                   "Precision (micro)", "Recall (micro)", "F1-Score (micro)"],
        "Value (%)": [ema_pct, top1_pct, micro_p, micro_r, micro_f1]
    })
    t2.to_excel(writer, sheet_name="Table II Overall", index=False)

    # Table III
    t3_rows = []
    for key, label in tier_labels.items():
        count = tier_counts[key]
        pct   = round(count / total * 100, 1)
        t3_rows.append({"Tier": label, "Queries Resolved": count, "% of Total": pct})
    pd.DataFrame(t3_rows).to_excel(writer, sheet_name="Table III Cascade", index=False)

    # Table IV
    t4 = pd.DataFrame({
        "Persona":   [p.capitalize() for p in PERSONAS],
        "Precision": [round(p_per[i]*100, 2)  for i in range(len(PERSONAS))],
        "Recall":    [round(r_per[i]*100, 2)  for i in range(len(PERSONAS))],
        "F1-Score":  [round(f1_per[i]*100, 2) for i in range(len(PERSONAS))],
        "Support":   [int(support[i])          for i in range(len(PERSONAS))]
    })
    t4.to_excel(writer, sheet_name="Table IV Per-Persona", index=False)

    # Table V
    t5_rows = []
    for baseline_persona in ["teacher", "friend"]:
        baseline_pred = [[baseline_persona]] * total
        b_true = to_binary(df["expected_list"], PERSONAS)
        b_pred = to_binary(baseline_pred,       PERSONAS)
        b_top1 = sum(1 for exp, pred in zip(df["expected_list"], baseline_pred) if set(pred) & set(exp))
        t5_rows.append({
            "System": f"Baseline — always '{baseline_persona}'",
            "Top-1 Accuracy (%)": round(b_top1 / total * 100, 2),
            "Precision (%)": round(precision_score(b_true, b_pred, average="micro", zero_division=0)*100, 2),
            "Recall (%)":    round(recall_score(b_true, b_pred,    average="micro", zero_division=0)*100, 2),
            "F1-Score (%)":  round(f1_score(b_true, b_pred,        average="micro", zero_division=0)*100, 2),
        })
    t5_rows.append({
        "System": "Router Prior (guides full multi-persona execution)",
        "Top-1 Accuracy (%)": top1_pct,
        "Precision (%)": micro_p,
        "Recall (%)":    micro_r,
        "F1-Score (%)":  micro_f1,
    })
    pd.DataFrame(t5_rows).to_excel(writer, sheet_name="Table V Baseline Comparison", index=False)

    # Misclassifications
    wrong.to_excel(writer, sheet_name="Misclassifications", index=False)

    # Full annotated report
    df.to_excel(writer, sheet_name="Full Annotated Report", index=False)

print(f"All tables saved to: {out_path}")
