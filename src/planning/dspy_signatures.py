import dspy
 
ollama_lm = dspy.OllamaLocal(
    model='mistral:7b-instruct',
    base_url='http://localhost:11434',
    max_tokens=1024,
)
dspy.settings.configure(lm=ollama_lm)
 
class QueryDecomposer(dspy.Signature):
    question      = dspy.InputField()
    sub_questions = dspy.OutputField()
    needs_web     = dspy.OutputField()
 
decomposer = dspy.ChainOfThought(QueryDecomposer)
