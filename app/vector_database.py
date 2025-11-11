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
_collection_name = "law_data"

# Initialize Qdrant client with error handling (#3 from task1.md)
try:
    print(f"INFO: Attempting to connect to Qdrant at {_qdrant_url}")
    _vector_database_client = QdrantClient(url=_qdrant_url)

    # Verify connection by getting collections
    _vector_database_client.get_collections()
    print("INFO: Successfully connected to Qdrant")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to connect to Qdrant at {_qdrant_url}: {e}")
    raise ConnectionError(f"Cannot connect to Qdrant database: {e}") from e

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
    try:
        if not vector_database_client.collection_exists(collection_name):
            print(f"INFO: Creating collection '{collection_name}' with vector size {vector_size}")
            vector_database_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"INFO: Collection '{collection_name}' created successfully")
        else:
            print(f"INFO: Collection '{collection_name}' already exists")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to create collection '{collection_name}': {e}")
        raise RuntimeError(f"Cannot create Qdrant collection: {e}") from e


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

    if not points or len(points) == 0:
        print("WARNING: No points to upload to Qdrant")
        return

    try:
        print(f"INFO: Uploading {len(points)} point(s) to collection '{collection_name}'")
        vector_database_client.upsert(
            collection_name=collection_name,
            points=points
        )

        # Verify upload succeeded (#5 from task1.md)
        collection_info = vector_database_client.get_collection(collection_name)
        points_count = collection_info.points_count

        if points_count == 0:
            raise RuntimeError(f"Upload verification failed: Collection '{collection_name}' has 0 points after upload")

        print(f"INFO: Upload complete - {points_count} point(s) now in collection '{collection_name}'")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to upload to Qdrant: {e}")
        raise RuntimeError(f"Cannot upload data to Qdrant: {e}") from e


def verify_embeddings_exist(vector_database_client: QdrantClient, collection_name: str) -> bool:
    """
    Verify that embeddings exist in the vector database (#4 from task1.md).
    Args:
        vector_database_client (QdrantClient): An instance of QdrantClient.
        collection_name (str): The name of the collection to check.
    Returns:
        bool: True if embeddings exist, raises exception otherwise.
    """
    try:
        # Check if collection exists
        if not vector_database_client.collection_exists(collection_name):
            raise RuntimeError(f"Collection '{collection_name}' does not exist in Qdrant")

        # Check if collection has any points
        collection_info = vector_database_client.get_collection(collection_name)
        points_count = collection_info.points_count

        if points_count == 0:
            raise RuntimeError(f"No embeddings found in collection '{collection_name}' (points count = 0)")

        print(f"INFO: Verified {points_count} embedding(s) exist in collection '{collection_name}'")
        return True
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to verify embeddings in vector database: {e}")
        raise RuntimeError(f"Cannot verify embeddings in database: {e}") from e


# lch_vector_store = QdrantVectorStore(client=_vector_database_client,
#                                     collection_name=_collection_name,
#                                     embedding=embeddings_for_qdrant_vector_store,
#                                     content_payload_key="text",)

def get_vector_store():
    """
    Create QdrantVectorStore lazily (only after collection exists) (#6 from task1.md).
    """
    try:
        print(f"INFO: Creating QdrantVectorStore for collection '{_collection_name}'")

        # Verify collection exists before creating vector store
        if not _vector_database_client.collection_exists(_collection_name):
            raise RuntimeError(f"Cannot create vector store: Collection '{_collection_name}' does not exist")

        vector_store = QdrantVectorStore(
            client=_vector_database_client,
            collection_name=_collection_name,
            embedding=embeddings_for_qdrant_vector_store,
            content_payload_key="text",
        )

        print("INFO: QdrantVectorStore created successfully")
        return vector_store
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to create QdrantVectorStore: {e}")
        raise RuntimeError(f"Cannot create vector store: {e}") from e

# def create_retriever():
#     """
#     Create a retriever from the Qdrant vector store.
#     Returns:
#         retriever: An instance of a retriever for similarity search.
#     """
#     retriever = lch_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#     return retriever

def create_retriever():
    """
    Create a retriever after data is loaded.
    """
    store = get_vector_store()
    return store.as_retriever(search_type="similarity", search_kwargs={"k": 1})



