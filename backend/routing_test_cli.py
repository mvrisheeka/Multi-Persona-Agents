import pandas as pd
import json
import asyncio
from router import classify_intent
from agent import supervisor_planner
from datetime import datetime
import random
import os

import router
router.ROUTER_EVAL_MODE = True
# ensure reports directory exists
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

EVAL_MODE = "router_prior"
EVAL_DESCRIPTION = (
    "Evaluates router prior quality against persona labels from the ground-truth Excel. "
    "It does not score the final synthesized answer."
)

# -------------------------------
# simulate supervisor (planning only)
# -------------------------------
def simulate_supervisor(query, personas):
    state = {
        "query": query,
        "intent_personas": personas
    }
    return asyncio.run(supervisor_planner(state))


# -------------------------------
# load Excel ground truth
# -------------------------------
def load_tests(xlsx_path):
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    df.columns = [c.strip().lower() for c in df.columns]

    tests = []
    for _, row in df.iterrows():
        query = str(row["query"])
        expected_raw = str(row["personas"]).lower()
        expected_personas = [p.strip() for p in expected_raw.split(",")]

        tests.append({
            "query": query,
            "expected": expected_personas
        })
    return tests


# -------------------------------
# evaluation + report
# -------------------------------
from datetime import datetime
import random


def evaluate(xlsx_path, sample_size=502):

    tests = load_tests(xlsx_path)

    # ---- random sampling ----
    if len(tests) > sample_size:
        tests = random.sample(tests, sample_size)

    total = len(tests)
    correct = 0

    report_rows = []

    for i, t in enumerate(tests, 1):
        query = t["query"]
        expected = t["expected"]

        # Router
        routing = classify_intent(query)
        personas = [p.lower() for p in routing["personas"]]

        # Supervisor
        plan = simulate_supervisor(query, personas)
        selected = plan["selected_personas"]
        priority = plan.get("priority_personas", personas)
        mode = plan["generation_mode"]

        # correctness
        overlap = set(personas) & set(expected)
        is_correct = len(overlap) > 0

        if is_correct:
            correct += 1

        report_rows.append({
            "evaluation_mode": EVAL_MODE,
            "evaluation_description": EVAL_DESCRIPTION,
            "query": query,
            "expected_personas": ", ".join(expected),
            "router_prior_personas": ", ".join(priority),
            "predicted_personas": ", ".join(personas),
            "execution_personas": ", ".join(selected),
            "generation_mode": mode,
            "correct": is_correct
        })

        status = "CORRECT" if is_correct else "WRONG"
        print(f"[{i}/{total}] {status} {query[:70]}")

    # ---- accuracy ----
    accuracy = round(correct / total * 100, 2)

    print("\n======================================")
    print("EVALUATION MODE:", EVAL_MODE)
    print("SAMPLED:", total)
    print("CORRECT:", correct)
    print("ACCURACY:", accuracy, "%")
    print("======================================")

    # ---- unique filename ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_name = os.path.join(REPORT_DIR, f"routing_report_{timestamp}.json")
    excel_name = os.path.join(REPORT_DIR, f"routing_report_{timestamp}.xlsx")

    # save JSON
    with open(json_name, "w", encoding="utf-8") as f:
        json.dump(report_rows, f, indent=2, ensure_ascii=False)

    # save Excel
    report_df = pd.DataFrame(report_rows)
    report_df.to_excel(excel_name, index=False)

    print("\nSaved files:")
    print(excel_name)
    print(json_name)


if __name__ == "__main__":
    evaluate("data/GroundTruth_test.xlsx")
