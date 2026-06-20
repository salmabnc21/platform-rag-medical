from langchain_community.llms import Ollama
from duckduckgo_search import DDGS
import re

llm = Ollama(
    model="cniongolo/biomistral",
    base_url="http://localhost:11434",
    temperature=0.1,
    top_p=0.9,
    top_k=40,
)


def planning_node(state):
    state["iteration"] = state.get("iteration", 0) + 1
    state["needs_web"] = False
    state["web_used"] = state.get("web_used", False)
    return state


def generation_node(state):
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    context = "\n".join([d.get("text", "") for d in docs]) if docs else ""
    context = context[:3000] if context else ""

    if context.strip():
        prompt = (
            "Tu es un assistant medical. Voici des donnees extraites d'un document "
            "(cela peut etre un texte normal ou un tableau de resultats d'analyse avec des "
            "valeurs numeriques, comme par exemple 'Hemoglobine 14,1 g/dl N:12-16').\n\n"
            "DONNEES DU DOCUMENT:\n"
            + context +
            "\n\nQUESTION: " + query +
            "\n\nInstructions:\n"
            "- Cherche attentivement la reponse dans les donnees ci-dessus, meme si elles sont "
            "presentees sous forme de tableau ou de liste de valeurs.\n"
            "- Si tu trouves la valeur demandee, donne-la clairement avec son unite.\n"
            "- Compare si possible avec les normes de reference indiquees (N: ...).\n"
            "- Si et SEULEMENT SI l'information demandee n'apparait nulle part dans les donnees "
            "ci-dessus, reponds exactement: INFORMATION ABSENTE\n\n"
            "Reponse:"
        )
    else:
        prompt = "Question: " + query + "\n\nReponds INFORMATION ABSENTE car aucun contexte disponible.\n\nReponse:"

    try:
        response = llm.invoke(prompt)
        response_text = response if isinstance(response, str) else str(response)

        if "Reponse:" in response_text:
            response_text = response_text.split("Reponse:", 1)[-1].strip()
        if "Réponse:" in response_text:
            response_text = response_text.split("Réponse:", 1)[-1].strip()

        lines = response_text.split("\n")
        clean_lines = []
        for line in lines:
            if line.strip():
                clean_lines.append(line.strip())
        response_text = "\n".join(clean_lines)

        state["answer"] = response_text
    except Exception as e:
        state["answer"] = "Erreur LLM: " + str(e)
        return state

    refus_signals = [
        "information absente", "absente", "je ne sais pas", "je ne peux pas",
        "pas d'information", "pas trouve", "n'existe pas", "cannot find",
        "no information", "ne sont pas mentionne", "ne sont pas directement mentionne",
        "n'ont pas ete identifie", "ne sont pas specifiquement mentionne",
        "pas mentionne", "ne contient pas", "n'est pas mentionne", "ne sont pas precise",
    ]

    response_lower = response_text.lower()
    is_refusal = False
    for signal in refus_signals:
        if signal in response_lower:
            is_refusal = True
            break
    if len(response_text.strip()) < 20:
        is_refusal = True

    if is_refusal and not state.get("web_used", False):
        state["needs_web"] = True
    else:
        state["needs_web"] = False

    return state


def web_search_node(state):
    query = state["query"]
    clean_query = re.sub(r"\b(quels?|quelles?|sont|les?|du|de|la|le)\b", "", query, flags=re.IGNORECASE)
    clean_query = re.sub(r"\s+", " ", clean_query).strip()
    search_query = clean_query + " medical symptomes"

    try:
        ddgs = DDGS()
        results = list(ddgs.text(search_query, max_results=3))

        if results:
            parts = []
            for r in results:
                title = r.get("title", "Sans titre")
                body = r.get("body", "")
                parts.append("Source: " + title + "\nContenu: " + body)
            web_context = "\n\n".join(parts)
            state["retrieved_docs"] = [{"text": web_context, "source": "web", "is_web": True}]
        else:
            state["retrieved_docs"] = [{"text": "Aucun resultat trouve sur le web.", "source": "web", "is_web": True}]

        state["web_used"] = True
        state["needs_web"] = False

    except Exception as e:
        state["retrieved_docs"] = [{"text": "Erreur recherche web: " + str(e), "source": "web", "is_web": True}]
        state["web_used"] = True
        state["needs_web"] = False

    return state
