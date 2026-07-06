import asyncio
from agent import graph

async def main():
    q = asyncio.Queue()
    config = {"configurable": {"token_queue": q}}
    state = {
        "query": "Explain quantum computing briefly",
        "selected_personas": ["teacher"],
        "priority_personas": ["teacher"]
    }
    
    # We will run the graph in a background task
    async def run_graph():
        try:
            async for event in graph.astream(state, config=config, stream_mode="updates"):
                print("[GRAPH UPDATE]", event)
            await q.put({"type": "done"})
        except Exception as e:
            print("[GRAPH ERROR]", e)
            await q.put({"type": "done"})
            
    asyncio.create_task(run_graph())
    
    while True:
        event = await q.get()
        if event["type"] == "done":
            break
        elif event["type"] == "token":
            print(f"[TOKEN] {event['id']}: {event['text']}", end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
