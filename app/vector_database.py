# Vector database
import os
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.documents import Document

import process_data

# Support both Docker and local setups
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "law_data"
VECTOR_DATABASE_CLIENT = QdrantClient(url=QDRANT_URL)

vector_size = len(process_data.embeddings[0]['embedding'])


def delete_collection_if_exists(vector_database_client: QdrantClient, collection_name: str):
    """
    Delete a collection in Qdrant if it exists.
    Args:
        vector_database_client (QdrantClient): An instance of QdrantClient.
        collection_name (str): The name of the collection to delete.
    """
    if vector_database_client.collection_exists(collection_name):
        vector_database_client.delete_collection(collection_name)
        print(f"'{collection_name}' DEL")


def create_collection_if_not_exists(vector_database_client: QdrantClient, collection_name: str, vector_size: int):
    """
    Create a collection in Qdrant if it does not exist.
    Args:
        vector_database_client (QdrantClient): An instance of QdrantClient.
        collection_name (str): The name of the collection to create.
        vector_size (int): The size of the vectors to be stored in the collection.
    """
    if not vector_database_client.collection_exists(collection_name):
        vector_database_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )


def create_points_from_embeddings(embeddings: list) -> list:
    """
    Create formatted points for Qdrant from embeddings with metadata.
    Args:
        embeddings (list): A list of dictionaries containing embeddings and metadata.
    Returns:
        list: A list of points formatted for Qdrant.
    """
    points = []
    for i, emb in enumerate(embeddings):
        payload = {
            "text": emb["text"],
            **emb["metadata"]
        }
        points.append(
            {
                "id": i,
                "vector": emb["embedding"].tolist(),
                "payload": payload,
            }
        )
    return points


def upload_to_qdrant(vector_database_client: QdrantClient, collection_name: str, points: list):
    """
    Upload embeddings to a Qdrant collection.
    Args:
        vector_database_client (QdrantClient): An instance of QdrantClient.
        collection_name (str): The name of the collection to upload to.
        points (list): A list of dictionaries containing embeddings and metadata.
    """

    vector_database_client.upsert(
        collection_name=collection_name,
        points=points
    )

    print("✅ Upload complete")


# delete_collection_if_exists(
#     vector_database_client=VECTOR_DATABASE_CLIENT,
#     collection_name=COLLECTION_NAME
# )

create_collection_if_not_exists(vector_database_client=VECTOR_DATABASE_CLIENT, collection_name=COLLECTION_NAME, vector_size=vector_size)

points = create_points_from_embeddings(embeddings=process_data.embeddings)
upload_to_qdrant(vector_database_client=VECTOR_DATABASE_CLIENT, collection_name=COLLECTION_NAME, points=points)

# vector_store = QdrantVectorStore(
#     client=vector_database_client,
#     collection_name="law_data",
# )
