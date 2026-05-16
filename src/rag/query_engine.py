from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from src.rag.indexing import chroma_client

# Récupérer la collection
collection = chroma_client.get_or_create_collection(
    name='medical_documents',
    metadata={'hnsw:space': 'cosine'}
)

# Création du QueryEngine
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode='compact',
)

def query_medical(question: str):
    response = query_engine.query(question)
    return str(response)