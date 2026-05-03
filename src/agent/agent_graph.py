from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import planning_node, generation_node, web_search_node

# Créer le graphe
workflow = StateGraph(AgentState)

# Ajouter les noeuds
workflow.add_node("plan",     planning_node)
workflow.add_node("generate", generation_node)
workflow.add_node("web",      web_search_node)

# Définir le flux
workflow.set_entry_point("plan")
workflow.add_conditional_edges(
    "plan",
    lambda s: "web" if s["needs_web"] else "generate",
    {"web": "web", "generate": "generate"}
)
workflow.add_edge("web",      "generate")
workflow.add_edge("generate", END)

# Compiler l'agent
app = workflow.compile() 
