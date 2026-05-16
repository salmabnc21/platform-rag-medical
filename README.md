# platform-rag-medical

Plateforme intelligente du l'Agentic RAG pour l’analyse de rapports médicaux PDF multimodaux



\# Plateforme Médicale Intelligente



\*\*Architecture Agentic RAG avec BioMistral\*\*



Système de question-réponse médical intelligent utilisant :

\- \*\*LangGraph\*\* : Orchestration de l'agent

\- \*\*LlamaIndex\*\* : Pipeline RAG

\- \*\*Chroma\*\* : Base vectorielle locale

\- \*\*BioMistral\*\* : LLM spécialisé médical



\## Installation



```bash

python -m venv rag-pfa

rag-pfa\\Scripts\\activate

pip install -r requirements.txt

```



\##  Démarrage



\*\*Terminal 1 :\*\*

```bash

ollama serve

```



\*\*Terminal 2 :\*\*

```bash

python main.py

```



\##  Répartition du Travail



\*\*Étudiant A : Agent LangGraph\*\*

\- src/agent/ (state, nodes, agent\_graph)



\*\*Étudiant B : Pipeline RAG\*\*

\- src/rag/ (ingestion, indexing, query\_engine)

\- src/graph/ (NER avec SpaCy)

