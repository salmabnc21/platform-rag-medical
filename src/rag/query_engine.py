from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from src.rag.indexing import chroma_client

collection = chroma_client.get_or_create_collection(
    name='medical_documents',
    metadata={'hnsw:space': 'cosine'}
)

vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode='tree_summarize',
)

def query_medical(question: str):
    response = query_engine.query(question)
    if not response or str(response).strip() == '':
        return "Aucune information trouvée dans les documents."
    return str(response)