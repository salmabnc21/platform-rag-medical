import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.agent_graph import app

result = app.invoke({
    "query": "Quels sont les traitements du diabète de type 2 ?",
    "retrieved_docs": [],
    "answer": "",
    "iteration": 0,
    "needs_web": False,
    "memory_context": ""
})

print("Réponse :", result["answer"])