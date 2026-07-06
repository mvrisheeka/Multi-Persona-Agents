import json
import os
import sys
import time
import asyncio
from agent import graph, CURRENT_MODEL_NAME
from datetime import datetime

# TRACING CONFIG
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "your-api-key-here"
# os.environ["LANGCHAIN_PROJECT"] = "multiagent-persona-lab"

# Ensure evals directory exists
EVALS_DIR = "data/evals"
os.makedirs(EVALS_DIR, exist_ok=True)

def type_text(text, delay=0.01):
    """Prints text with a typing animation effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def save_eval_log(inputs, final_state, latency):
    """Saves an industrial-grade evaluation JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_latency_seconds": round(latency, 2),
            "generation_latency": round(final_state.get("gen_latency", 0), 2),
            "judge_latency": round(final_state.get("judge_latency", 0), 2),
            "models": {
                "generator": CURRENT_MODEL_NAME,
                "judge": CURRENT_MODEL_NAME
            }
        },
        "input": inputs,
        "routing": {
            "priority_personas": final_state.get("priority_personas"),
            "intent_personas": final_state.get("intent_personas"),
            "selected_personas": final_state.get("selected_personas")
        },
        "evaluation": {
            "confidence_score": final_state.get("confidence"),
            "confidence_label": final_state.get("confidence_label"),
            "judge_rationale": final_state.get("judge_rationale"),
            "generation_mode": final_state.get("generation_mode")
        },
        "metrics": {
            "total_latency_seconds": round(latency, 2),
            "generation_latency": round(final_state.get("gen_latency", 0), 2),
            "judge_latency": round(final_state.get("judge_latency", 0), 2),
            "answer_char_count": len(final_state.get("answer", "")),
            "node_details": final_state.get("node_metrics", {}),
            "persona_outputs_summary": {
                persona: {"length": len(text)} 
                for persona, text in final_state.get("persona_outputs", {}).items()
            }
        },
        "final_answer": final_state.get("answer")
    }
    
    filename = os.path.join(EVALS_DIR, f"eval_{timestamp}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)
    return filename

def main():
    print("\033[94m" + "="*50 + "\033[0m")
    print("\033[94m      TERMINAL MULTI-AGENT CHAT (LOCAL)      \033[0m")
    print("\033[94m" + "="*50 + "\033[0m")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("\033[92mYou:\033[0m ").strip()
            
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                break

            print("\n" + "\033[90m" + "-"*30 + "\033[0m")
            print("\033[90mProcessing query...\033[0m")
            
            start_time = time.time()
            inputs = {"query": query}

            async def run_query(inputs):
                final_state_local = {}
                printed_nodes = set()
                async for output in graph.astream(inputs, stream_mode="updates"):
                    for node_name, state in output.items():
                        if isinstance(state, dict):
                            if "node_metrics" in state and "node_metrics" in final_state_local:
                                final_state_local["node_metrics"].update(state["node_metrics"])
                            final_state_local.update(state)

                        if node_name not in printed_nodes:
                            if node_name == "router":
                                print(f"\033[93m[ROUTER]\033[0m Identified personas: {state.get('intent_personas')}")
                            elif node_name == "supervisor":
                                print(f"\033[93m[SUPERVISOR]\033[0m Task type: {state.get('generation_mode')} mode")
                                print(f"\033[93m[DISPATCH]\033[0m -> {state.get('selected_personas')}")
                            printed_nodes.add(node_name)

                return final_state_local

            final_state = asyncio.run(run_query(inputs))
            latency = time.time() - start_time

            # Final evaluation and logging
            latency = time.time() - start_time
            
            # Fix: Handle 1-10 score correctly for percentage display
            raw_score = final_state.get('confidence', 0)
            display_score = raw_score * 10 if raw_score <= 10 else raw_score
            
            print(f"\n\033[96m[JUDGE VERDICT] {final_state.get('confidence_label', 'N/A')} ({display_score:.0f}%)\033[0m")
            print(f"\033[96mRationale: {final_state.get('judge_rationale', 'No rationale provided.')}\033[0m")
            
            gen_lat = final_state.get("gen_latency", 0)
            jud_lat = final_state.get("judge_latency", 0)
            print(f"\033[90m[METRICS] Gen: {gen_lat:.2f}s | Judge: {jud_lat:.2f}s | Total: {latency:.2f}s\033[0m")

            eval_file = save_eval_log(inputs, final_state, latency)
            print(f"\n\033[90m[EVAL LOG] Saved to: {eval_file}\033[0m")
            print("\033[90m" + "-"*30 + "\033[0m")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n\033[91m[ERROR] An error occurred: {e}\033[0m")

if __name__ == "__main__":
    main()
