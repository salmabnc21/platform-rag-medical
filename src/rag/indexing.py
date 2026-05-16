import chromadb
from chromadb.config import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings as LlamaSettings

# Configuration Chroma
chroma_client = chromadb.PersistentClient(
    path='./data/chroma_db',
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_or_create_collection(
    name='medical_documents',
    metadata={'hnsw:space': 'cosine'}
)

# Modèle d'embedding
LlamaSettings.embed_model = HuggingFaceEmbedding(
    model_name='BAAI/bge-large-en-v1.5',
    cache_folder='./data/models',
)
from llama_index.llms.ollama import Ollama

# Modèle LLM biomistral
LlamaSettings.llm = Ollama(
    model='cniongolo/biomistral',
    base_url='http://localhost:11434',
    request_timeout=120.0,
)