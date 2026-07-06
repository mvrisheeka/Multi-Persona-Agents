from agent import graph
import os

def visualize():
    # Attempt to draw the graph as a PNG
    try:
        graph_img = graph.get_graph().draw_mermaid_png()
        filename = "agent_architecture.png"
        with open(filename, "wb") as f:
            f.write(graph_img)
        print(f"Success! Graph visualization saved to: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"Could not generate PNG (likely missing pygraphviz or mermaid dependencies): {e}")
        print("\nFallback: Printing Mermaid diagram text below. You can paste this into https://mermaid.live/")
        print("-" * 50)
        print(graph.get_graph().draw_mermaid())
        print("-" * 50)

if __name__ == "__main__":
    visualize()
