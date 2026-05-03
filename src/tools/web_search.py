from duckduckgo_search import DDGS
from langchain.tools import tool
 
@tool
def web_search(query: str, max_results: int = 5) -> list:
    '''Recherche web sans clé API — DuckDuckGo'''
    with DDGS() as ddgs:
        return [r for r in ddgs.text(query, max_results=max_results)]
