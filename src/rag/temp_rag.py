import uuid
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        _embed_model = HuggingFaceEmbedding(
            model_name='BAAI/bge-large-en-v1.5',
            cache_folder='./data/models',
        )
    return _embed_model


def build_temp_index(text: str):
    """Chunk text and index it in an ephemeral in-memory ChromaDB collection.

    Returns (client, index, n_chunks). The caller owns the client lifetime -
    the collection is destroyed automatically when the client is
    garbage-collected (EphemeralClient never writes to disk).
    """
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=30)
    nodes = splitter.get_nodes_from_documents([Document(text=text)])

    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name=f"temp_{uuid.uuid4().hex[:8]}",
        metadata={"hnsw:space": "cosine"},
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=_get_embed_model(),
    )
    return client, index, len(nodes)


def query_temp_index(index, query: str, top_k: int = 3) -> str:
    """Return the top_k most relevant chunks from a temporary index."""
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return "\n\n".join(n.get_content() for n in nodes)
