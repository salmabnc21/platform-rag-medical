from typing import TypedDict, List, Optional
 
class AgentState(TypedDict):
    query:           str
    sub_questions:   List[str]
    retrieved_docs:  List[dict]
    web_results:     List[dict]
    answer:          str
    ragas_scores:    dict
    iteration:       int
    needs_web:       bool
    memory_context:  str

