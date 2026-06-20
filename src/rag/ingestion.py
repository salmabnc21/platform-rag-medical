import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from src.rag.indexing import collection

# Chargement des documents médicaux
documents = SimpleDirectoryReader(
    input_dir='./data/documents/',
    required_exts=['.pdf', '.docx', '.txt'],
).load_data()

print("=" * 50)
print("NB DOCUMENTS =", len(documents))

for i, doc in enumerate(documents):
    print(f"DOC {i+1} :", doc.metadata)

print("=" * 50)

# Chunking sémantique
parser = SentenceWindowNodeParser.from_defaults(window_size=3)
nodes  = parser.get_nodes_from_documents(documents)

# Indexation dans Chroma
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_ctx  = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_ctx) 
