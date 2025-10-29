# Vector database
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

import process_data

vector_database_client = QdrantClient(url="http://localhost:6333")

vector_size = len(process_data.embeddings[0]['embedding'])


if vector_database_client.collection_exists("law_data"):
    vector_database_client.delete_collection("law_data")
    print("'law_data' DEL")

if not vector_database_client.collection_exists("law_data"):
    vector_database_client.create_collection(
        collection_name="law_data",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

vector_store = QdrantVectorStore(
    client=vector_database_client,
    collection_name="law_data",
)
