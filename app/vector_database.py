# Vector database
import os
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.documents import Document

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings

embeddings_for_qdrant_vector_store = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

import process_data

# Support both Docker and local setups
_qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
_vector_database_client = QdrantClient(url=_qdrant_url)

# vector_size = len(process_data.embeddings[0]['embedding'])


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


# lch_vector_store = QdrantVectorStore(client=_vector_database_client,
#                                     collection_name=_collection_name,
#                                     embedding=embeddings_for_qdrant_vector_store,
#                                     content_payload_key="text",)

def get_vector_store(collection_name: str) -> QdrantVectorStore:
    """
    Create QdrantVectorStore lazily (only after collection exists).
    """
    return QdrantVectorStore(
        client=_vector_database_client,
        collection_name=collection_name,
        embedding=embeddings_for_qdrant_vector_store,
        content_payload_key="text",
    )

# def create_retriever():
#     """
#     Create a retriever from the Qdrant vector store.
#     Returns:
#         retriever: An instance of a retriever for similarity search.
#     """
#     retriever = lch_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#     return retriever

def create_retriever(collection_name: str):
    """
    Create a retriever after data is loaded.
    """
    store = get_vector_store(collection_name)
    return store.as_retriever(search_type="similarity", search_kwargs={"k": 1})



