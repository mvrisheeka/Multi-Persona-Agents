from typing import Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import asyncio
import time
from agent import graph, ALL_PERSONAS, CURRENT_MODEL_NAME
import os
from datetime import datetime

app = FastAPI()

# Simple in-memory session for focused "chat more" context
SESSION_MEMORY = {
    "last_query": "",
    "last_outputs": {},
    "persona_threads": {},
    "last_combined_output": ""
}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EVALS_DIR = "data/evals"
os.makedirs(EVALS_DIR, exist_ok=True)

def _estimate_tokens(text: str) -> int:
    """Lightweight token estimate for logging (roughly chars/4)."""
    if not text:
        return 0
    return max(1, int(round(len(text) / 4)))

def _build_confidence_math(final_state: dict) -> dict:
    """
    Reconstruct a transparent confidence calculation breakdown for eval logs.
    Mirrors the current weighted scheme used in the agent pipeline.
    """
    outputs = final_state.get("persona_outputs", {}) or {}
    selected = final_state.get("selected_personas", []) or []
    priority = final_state.get("priority_personas", []) or []
    confidence = final_state.get("confidence", 0.0) or 0.0

    # Rebuild per-persona proxy scores from available final outputs.
    per_persona_scores = {}
    for persona, text in outputs.items():
        score = 0.2
        if persona in priority:
            try:
                idx = priority.index(persona)
            except ValueError:
                idx = 99
            score += max(0.3 - (0.1 * idx), 0.1)
        if len((text or "").split()) >= 25:
            score += 0.15
        per_persona_scores[persona] = round(min(score, 1.0), 3)

    avg_score = (sum(per_persona_scores.values()) / len(per_persona_scores)) if per_persona_scores else 0.0
    matched_priority = len(set(priority) & set(per_persona_scores))
    priority_coverage = (matched_priority / max(1, len(priority))) if priority else 1.0
    participation = len(per_persona_scores) / max(1, len(selected))

    base = 0.55
    w_avg = 0.2 * avg_score
    w_priority = 0.15 * priority_coverage
    w_participation = 0.1 * participation
    recomputed = round(min(base + w_avg + w_priority + w_participation, 0.99), 2)

    return {
        "formula": "min(0.55 + 0.2*avg_score + 0.15*priority_coverage + 0.1*participation, 0.99)",
        "inputs": {
            "selected_personas_count": len(selected),
            "priority_personas_count": len(priority),
            "outputs_count": len(per_persona_scores),
            "avg_score": round(avg_score, 4),
            "priority_coverage": round(priority_coverage, 4),
            "participation": round(participation, 4),
            "per_persona_scores": per_persona_scores,
        },
        "terms": {
            "base": base,
            "weighted_avg_score": round(w_avg, 4),
            "weighted_priority_coverage": round(w_priority, 4),
            "weighted_participation": round(w_participation, 4),
        },
        "recomputed_confidence": recomputed,
        "final_state_confidence": round(float(confidence), 2),
        "delta": round(float(confidence) - recomputed, 4),
    }

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
        "priority_storage": {
            "primary_persona": (final_state.get("priority_personas") or ["teacher"])[0],
            "reviewed_priority_output": final_state.get("reviewed_priority_output", "")
        },
        "evaluation": {
            "confidence_score": final_state.get("confidence"),
            "confidence_label": final_state.get("confidence_label"),
            "judge_rationale": final_state.get("judge_rationale"),
            "generation_mode": final_state.get("generation_mode"),
            "confidence_math": _build_confidence_math(final_state)
        },
        "metrics": {
            "total_latency_seconds": round(latency, 2),
            "generation_latency": round(final_state.get("gen_latency", 0), 2),
            "judge_latency": round(final_state.get("judge_latency", 0), 2),
            "answer_char_count": len(final_state.get("answer", "")),
            "answer_token_estimate": _estimate_tokens(final_state.get("answer", "")),
            "node_details": final_state.get("node_metrics", {}),
            "persona_outputs_summary": {
                persona: {
                    "length": len(text),
                    "token_estimate": _estimate_tokens(text),
                    "word_count": len((text or "").split()),
                    "content": text
                } 
                for persona, text in final_state.get("persona_outputs", {}).items()
            },
            "token_estimates": {
                "query_tokens": _estimate_tokens((inputs or {}).get("query", "")),
                "all_persona_output_tokens": sum(
                    _estimate_tokens(t) for t in (final_state.get("persona_outputs", {}) or {}).values()
                ),
                "final_answer_tokens": _estimate_tokens(final_state.get("answer", "")),
                "reviewed_priority_output_tokens": _estimate_tokens(final_state.get("reviewed_priority_output", "")),
            },
        },
        "final_answer": final_state.get("answer")
    }
    
    filename = os.path.join(EVALS_DIR, f"eval_{timestamp}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)
    return filename

@app.get("/chat/stream")
async def chat_stream(query: str, request: Request, focus: Optional[str] = None):
    async def event_generator():
        start_time = time.time()
        # Unified queue for all events (tokens, state updates, etc.)
        queue = asyncio.Queue()
        final_emitted = False
        
        # Handle focused "chat more" context
        actual_query = query
        if focus:
            thread = SESSION_MEMORY["persona_threads"].setdefault(focus, [])

            if thread:
                transcript = "\n".join(
                    [f"{turn['role'].upper()}: {turn['content']}" for turn in thread]
                )
                actual_query = (
                    f"You are continuing an ongoing 1:1 chat with the {focus} persona.\n\n"
                    f"Conversation so far:\n{transcript}\n\n"
                    f"User's new message:\n{query}\n\n"
                    "Respond naturally as the same persona, consistent with prior context."
                )
            elif focus == "combined" and SESSION_MEMORY.get("last_combined_output"):
                actual_query = (
                    f"Original Context: {SESSION_MEMORY['last_query']}\n\n"
                    f"Previous Combined Answer: {SESSION_MEMORY['last_combined_output']}\n\n"
                    f"User Request: {query}\n"
                    "Continue this same combined conversation naturally."
                )
            elif focus in SESSION_MEMORY["last_outputs"]:
                # First focused turn can bootstrap from the last multi-agent output.
                previous_output = SESSION_MEMORY["last_outputs"][focus]
                actual_query = (
                    f"Original Context: {SESSION_MEMORY['last_query']}\n\n"
                    f"Your Previous Output: {previous_output}\n\n"
                    f"User Request: {query}\n"
                    "Provide a deep, highly focused continuation based strictly on the above context."
                )
        else:
            SESSION_MEMORY["last_query"] = query

        initial_state = {
            "query": actual_query,
            "focus_persona": focus,
            "persona_outputs": {},
            "selected_personas": [],
            "priority_personas": ["teacher"],
            "generation_mode": "medium"
        }
        
        # Helper to run the graph in background
        async def run_graph():
            try:
                # Use stream_mode="updates" to get individual node outputs
                # Pass token_queue via config sidecar for zero-latency streaming
                config = {"configurable": {"token_queue": queue}}
                async for event in graph.astream(initial_state, config=config, stream_mode="updates"):
                    await queue.put({"type": "graph_event", "data": event})
                await queue.put({"type": "done"}) # Signal completion
            except Exception as e:
                print(f"[GRAPH ERROR] {e}")
                await queue.put({"type": "error", "message": str(e)})

        graph_task = asyncio.create_task(run_graph())
        
        try:
            while True:
                if await request.is_disconnected():
                    graph_task.cancel()
                    break
                
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # Periodically send a heartbeat log to keep the UI/connection alive
                    if time.time() - start_time > 10 and not focus:
                         heartbeat = json.dumps({'type': 'log', 'message': 'Experts are still processing... thanks for your patience.'})
                         yield f"data: {heartbeat}\n\n"
                    continue

                if event["type"] == "done":
                    # Safety fallback: if combine path didn't emit final, send best available answer.
                    if not final_emitted:
                        fallback_answer = initial_state.get("answer", "")
                        if not fallback_answer and focus:
                            fallback_answer = initial_state.get("persona_outputs", {}).get(focus, "")
                        if not fallback_answer and initial_state.get("persona_outputs"):
                            fallback_answer = next(iter(initial_state["persona_outputs"].values()))
                        if fallback_answer:
                            elapsed = time.time() - start_time
                            yield f"data: {json.dumps({'type': 'final', 'answer': fallback_answer, 'confidence': initial_state.get('confidence', 1.0), 'priority_personas': initial_state.get('priority_personas', ['teacher']), 'latency': elapsed, 'generation_mode': initial_state.get('generation_mode'), 'selected_personas': initial_state.get('selected_personas', []), 'node_metrics': initial_state.get('node_metrics', {}), 'gen_latency': initial_state.get('gen_latency', 0.0), 'judge_latency': initial_state.get('judge_latency', 0.0), 'fallback': True})}\n\n"
                    break
                if event["type"] == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': event['message']})}\n\n"
                    break
                
                if event["type"] == "token":
                    yield f"data: {json.dumps({'type': 'token', 'id': event['id'], 'text': event['text']})}\n\n"
                
                if event["type"] == "graph_event":
                    data = event.get("data")
                    if not data or not isinstance(data, dict):
                        continue

                    for node_name, output in data.items():
                        if not output: continue
                        initial_state.update(output)
                        elapsed = time.time() - start_time

                        # Emit routing/priorities as soon as supervisor decides
                        if node_name == "supervisor" and (
                            "priority_personas" in output or "selected_personas" in output
                        ):
                            routing_msg = (
                                f"[{elapsed:.1f}s] Routing decided. "
                                f"Priority={output.get('priority_personas', [])} | "
                                f"Selected={output.get('selected_personas', [])}"
                            )
                            yield f"data: {json.dumps({'type': 'log', 'message': routing_msg})}\n\n"
                            yield f"data: {json.dumps({'type': 'routing', 'priority_personas': output.get('priority_personas', []), 'selected_personas': output.get('selected_personas', [])})}\n\n"

                        # When a persona is fully done, send the final text (includes sources/invites)
                        if "persona_outputs" in output and node_name in ALL_PERSONAS:
                            for p, txt in output["persona_outputs"].items():
                                SESSION_MEMORY["last_outputs"][p] = txt
                                yield f"data: {json.dumps({'type': 'log', 'message': f'[{elapsed:.1f}s] Persona complete: {p} ({len(txt)} chars)'})}\n\n"
                                yield f"data: {json.dumps({'type': 'persona', 'id': p, 'text': txt})}\n\n"

                        # Emit node metrics live when available
                        if "node_metrics" in output and isinstance(output["node_metrics"], dict):
                            for metric_node, metric_payload in output["node_metrics"].items():
                                yield f"data: {json.dumps({'type': 'metric', 'node': metric_node, 'payload': metric_payload, 'elapsed': round(elapsed, 2)})}\n\n"
                        
                        # Final synthesis
                        if "answer" in output and node_name == "combine":
                            if focus:
                                thread = SESSION_MEMORY["persona_threads"].setdefault(focus, [])
                                thread.append({"role": "user", "content": query})
                                thread.append({"role": "assistant", "content": output["answer"]})
                                # Keep recent turns only to avoid unbounded growth.
                                if len(thread) > 16:
                                    SESSION_MEMORY["persona_threads"][focus] = thread[-16:]
                            SESSION_MEMORY["last_combined_output"] = output["answer"]
                            yield f"data: {json.dumps({
                                'type': 'final', 
                                'answer': output['answer'], 
                                'confidence': initial_state.get('confidence', 1.0),
                                'priority_personas': initial_state.get('priority_personas', ['teacher']),
                                'latency': elapsed,
                                'generation_mode': initial_state.get('generation_mode'),
                                'selected_personas': initial_state.get('selected_personas', []),
                                'node_metrics': initial_state.get('node_metrics', {}),
                                'gen_latency': initial_state.get('gen_latency', 0.0),
                                'judge_latency': initial_state.get('judge_latency', 0.0)
                            })}\n\n"
                            final_emitted = True
                            
                        # Logs for the UI
                        yield f"data: {json.dumps({'type': 'log', 'message': f'[{elapsed:.1f}s] Step finished: {node_name}'})}\n\n"

        except Exception as e:
            print(f"[STREAM ERROR] {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            # Final Cleanup & Save Log
            latency = time.time() - start_time
            try:
                save_eval_log({"query": query}, initial_state, latency)
            except: pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    print("\033[94m" + "="*50 + "\033[0m")
    print("\033[94m   MULTI-AGENT PERSONA STUDIO - SERVER LOADED    \033[0m")
    print("\033[94m" + "="*50 + "\033[0m")
    print("\033[92m[READY]\033[0m Try these example prompts:")
    print(" - \033[93mTeacher Focus:\033[0m 'Explain quantum entanglement with code examples.'")
    print(" - \033[93mCounselor Focus:\033[0m 'I feel overwhelmed by my workload lately.'")
    print(" - \033[93mParent Focus:\033[0m 'How do I build a consistent study routine?'")
    print(" - \033[93mFriend Focus:\033[0m 'Yo, recommend some chill music for coding.'")
    print("\033[90m" + "-"*50 + "\033[0m\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
