from langchain_community.llms import Ollama
from duckduckgo_search import DDGS

# Connexion à BioMistral
llm = Ollama(
    model="cniongolo/biomistral",
    base_url="http://localhost:11434",
    temperature=0.1,
)

def planning_node(state):
    """Analyse la question et décide si recherche web nécessaire"""
    query = state["query"]
    keywords = ["récent", "actualité", "nouveau", "2024", "2025"]
    state["needs_web"] = any(k in query.lower() for k in keywords)
    state["iteration"] = state.get("iteration", 0) + 1
    return state

def generation_node(state):
    """BioMistral génère la réponse médicale"""
    query   = state["query"]
    docs    = state.get("retrieved_docs", [])
    context = "\n".join([d.get("text", "") for d in docs])

    prompt = f"""Tu es un assistant médical expert.
Contexte médical : {context}
Question : {query}
Réponds en français de manière précise et médicale."""

    state["answer"] = llm.invoke(prompt)
    return state

def web_search_node(state):
    """Recherche web DuckDuckGo si nécessaire"""
    with DDGS() as ddgs:
        results = list(ddgs.text(state["query"], max_results=3))
    state["retrieved_docs"] = [
        {"text": r["body"], "source": r["href"]} for r in results
    ]
    return state 
