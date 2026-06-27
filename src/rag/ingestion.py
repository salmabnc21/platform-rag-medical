import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from src.rag.indexing import collection

documents = SimpleDirectoryReader(
    input_dir='./data/documents/',
    required_exts=['.pdf', '.docx', '.txt'],
).load_data()

print("=" * 50)
print("NB DOCUMENTS =", len(documents))
print("=" * 50)

parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)

print("=" * 50)
print("NB CHUNKS GENERES =", len(nodes))
print("=" * 50)

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_ctx)

print("=" * 50)
print("INDEXATION TERMINEE")
print("Nombre total de chunks dans la base:", collection.count())
print("=" * 50)
