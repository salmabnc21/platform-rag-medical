from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import planning_node, generation_node, web_search_node

def build_agent_graph():
    """Construire le graphe agent"""
    workflow = StateGraph(AgentState)

    workflow.add_node("plan", planning_node)
    workflow.add_node("generate", generation_node)
    workflow.add_node("web", web_search_node)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "generate")

    def should_use_web(state):
        if state.get("needs_web", False) and not state.get("web_used", False):
            print("[nodes.py] Transition -> WEB_SEARCH")
            return "web"
        print("[nodes.py] Transition -> END")
        return "end"

    workflow.add_conditional_edges(
        "generate",
        should_use_web,
        {
            "web": "web",
            "end": END
        }
    )

    workflow.add_edge("web", "generate")

    return workflow.compile()

app = build_agent_graph()