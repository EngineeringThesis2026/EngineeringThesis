# Vector database
import os
from typing import Optional
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.documents import Document

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings

from exceptions import QdrantConnectionError, QdrantOperationError, EmbeddingModelError
from logger import logger

import process_data

# Support both Docker and local setups
_qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
_collection_name = "law_data"
_vector_database_client = None
_embeddings_for_qdrant_vector_store = None


def get_qdrant_embeddings():
    """
    Get or create HuggingFace embeddings instance for Qdrant (singleton pattern).

    Returns:
        HuggingFaceEmbeddings: The embeddings instance.

    Raises:
        EmbeddingModelError: If embeddings model fails to load.
    """
    global _embeddings_for_qdrant_vector_store

    if _embeddings_for_qdrant_vector_store is None:
        try:
            logger.info("Initializing HuggingFace embeddings for Qdrant")
            _embeddings_for_qdrant_vector_store = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Successfully initialized HuggingFace embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embeddings: {type(e).__name__}: {e}")
            raise EmbeddingModelError(f"Failed to initialize embeddings for Qdrant: {e}")

    return _embeddings_for_qdrant_vector_store


def get_qdrant_client() -> QdrantClient:
    """
    Get or create Qdrant client instance (singleton pattern).

    Returns:
        QdrantClient: The Qdrant client instance.

    Raises:
        QdrantConnectionError: If connection to Qdrant fails.
    """
    global _vector_database_client

    if _vector_database_client is None:
        try:
            logger.info(f"Connecting to Qdrant at: {_qdrant_url}")
            _vector_database_client = QdrantClient(url=_qdrant_url)

            # Test connection
            _vector_database_client.get_collections()
            logger.info("Successfully connected to Qdrant")

        except ConnectionError as e:
            logger.error(f"Connection error to Qdrant at {_qdrant_url}: {e}")
            raise QdrantConnectionError(f"Cannot connect to Qdrant at {_qdrant_url}. Ensure Qdrant is running: {e}")

        except Exception as e:
            logger.error(f"Unexpected error connecting to Qdrant: {type(e).__name__}: {e}")
            raise QdrantConnectionError(f"Failed to connect to Qdrant at {_qdrant_url}: {e}")

    return _vector_database_client

# vector_size = len(process_data.embeddings[0]['embedding'])


def delete_collection_if_exists(vector_database_client: QdrantClient, collection_name: str):
    """
    Delete a collection in Qdrant if it exists.

    Args:
        vector_database_client (QdrantClient): An instance of QdrantClient.
        collection_name (str): The name of the collection to delete.

    Raises:
        QdrantOperationError: If deletion fails.
    """
    try:
        if vector_database_client.collection_exists(collection_name):
            logger.info(f"Deleting existing collection: {collection_name}")
            vector_database_client.delete_collection(collection_name)
            logger.info(f"Successfully deleted collection: {collection_name}")
        else:
            logger.info(f"Collection '{collection_name}' does not exist, skipping deletion")

    except ConnectionError as e:
        logger.error(f"Connection lost while deleting collection: {e}")
        raise QdrantOperationError(f"Lost connection to Qdrant while deleting collection: {e}")

    except Exception as e:
        logger.error(f"Error deleting collection '{collection_name}': {type(e).__name__}: {e}")
        raise QdrantOperationError(f"Failed to delete collection '{collection_name}': {e}")


def create_collection_if_not_exists(vector_database_client: QdrantClient, collection_name: str, vector_size: int):
    """
    Create a collection in Qdrant if it does not exist.

    Args:
        vector_database_client (QdrantClient): An instance of QdrantClient.
        collection_name (str): The name of the collection to create.
        vector_size (int): The size of the vectors to be stored in the collection.

    Raises:
        QdrantOperationError: If collection creation fails.
    """
    try:
        if not vector_database_client.collection_exists(collection_name):
            logger.info(f"Creating collection '{collection_name}' with vector size {vector_size}")
            vector_database_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Successfully created collection: {collection_name}")
        else:
            logger.info(f"Collection '{collection_name}' already exists, skipping creation")

    except ValueError as e:
        logger.error(f"Invalid parameters for collection creation: {e}")
        raise QdrantOperationError(f"Invalid vector parameters (size={vector_size}): {e}")

    except ConnectionError as e:
        logger.error(f"Connection lost while creating collection: {e}")
        raise QdrantOperationError(f"Lost connection to Qdrant while creating collection: {e}")

    except Exception as e:
        logger.error(f"Error creating collection '{collection_name}': {type(e).__name__}: {e}")
        raise QdrantOperationError(f"Failed to create collection '{collection_name}': {e}")


def create_points_from_embeddings(embeddings: list) -> list:
    """
    Create formatted points for Qdrant from embeddings with metadata.

    Args:
        embeddings (list): A list of dictionaries containing embeddings and metadata.

    Returns:
        list: A list of points formatted for Qdrant.

    Raises:
        QdrantOperationError: If point creation fails due to invalid data structure.
    """
    try:
        if not embeddings:
            logger.error("Cannot create points from empty embeddings list")
            raise QdrantOperationError("No embeddings provided for point creation")

        logger.info(f"Creating {len(embeddings)} point(s) for Qdrant upload")

        points = []
        for i, emb in enumerate(embeddings):
            # Validate embedding structure
            if "embedding" not in emb:
                raise KeyError(f"Embedding at index {i} missing 'embedding' key")
            if "text" not in emb:
                raise KeyError(f"Embedding at index {i} missing 'text' key")
            if "metadata" not in emb:
                raise KeyError(f"Embedding at index {i} missing 'metadata' key")

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

        logger.info(f"Successfully created {len(points)} point(s)")
        return points

    except KeyError as e:
        logger.error(f"Invalid embedding structure: {e}")
        raise QdrantOperationError(f"Invalid embedding data structure: {e}")

    except AttributeError as e:
        logger.error(f"Embedding has no tolist() method: {e}")
        raise QdrantOperationError(f"Invalid embedding format (not a numpy array): {e}")

    except TypeError as e:
        logger.error(f"Type error creating points: {e}")
        raise QdrantOperationError(f"Invalid data types in embeddings: {e}")

    except Exception as e:
        logger.error(f"Unexpected error creating points: {type(e).__name__}: {e}")
        raise QdrantOperationError(f"Failed to create points from embeddings: {e}")


def upload_to_qdrant(vector_database_client: QdrantClient, collection_name: str, points: list):
    """
    Upload embeddings to a Qdrant collection.

    Args:
        vector_database_client (QdrantClient): An instance of QdrantClient.
        collection_name (str): The name of the collection to upload to.
        points (list): A list of dictionaries containing embeddings and metadata.

    Raises:
        QdrantOperationError: If upload fails.
    """
    try:
        if not points:
            logger.error("Cannot upload empty points list")
            raise QdrantOperationError("No points provided for upload")

        logger.info(f"Uploading {len(points)} point(s) to collection '{collection_name}'")

        vector_database_client.upsert(
            collection_name=collection_name,
            points=points
        )

        logger.info(f"Successfully uploaded {len(points)} point(s) to Qdrant")

    except ConnectionError as e:
        logger.error(f"Connection lost during upload: {e}")
        raise QdrantOperationError(f"Lost connection to Qdrant during upload: {e}")

    except TimeoutError as e:
        logger.error(f"Upload timeout: {e}")
        raise QdrantOperationError(f"Upload to Qdrant timed out: {e}")

    except ValueError as e:
        logger.error(f"Invalid points format for upload: {e}")
        raise QdrantOperationError(f"Invalid points data for upload: {e}")

    except MemoryError as e:
        logger.error(f"Out of memory during upload: {e}")
        raise QdrantOperationError(f"Out of memory - try uploading fewer points at once: {e}")

    except Exception as e:
        logger.error(f"Unexpected error uploading to Qdrant: {type(e).__name__}: {e}")
        raise QdrantOperationError(f"Failed to upload embeddings to Qdrant: {e}")


# lch_vector_store = QdrantVectorStore(client=_vector_database_client,
#                                     collection_name=_collection_name,
#                                     embedding=embeddings_for_qdrant_vector_store,
#                                     content_payload_key="text",)

def get_vector_store():
    """
    Create QdrantVectorStore lazily (only after collection exists).

    Returns:
        QdrantVectorStore: The vector store instance.

    Raises:
        QdrantOperationError: If vector store initialization fails.
    """
    try:
        logger.info("Initializing QdrantVectorStore")

        # Get client and embeddings (these will use singleton pattern)
        client = get_qdrant_client()
        embeddings = get_qdrant_embeddings()

        store = QdrantVectorStore(
            client=client,
            collection_name=_collection_name,
            embedding=embeddings,
            content_payload_key="text",
        )

        logger.info("Successfully initialized QdrantVectorStore")
        return store

    except (QdrantConnectionError, EmbeddingModelError) as e:
        # Re-raise our custom exceptions
        logger.error(f"Failed to initialize vector store: {e}")
        raise

    except ValueError as e:
        logger.error(f"Invalid configuration for vector store: {e}")
        raise QdrantOperationError(f"Invalid vector store configuration: {e}")

    except Exception as e:
        logger.error(f"Unexpected error initializing vector store: {type(e).__name__}: {e}")
        raise QdrantOperationError(f"Failed to initialize vector store: {e}")

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

    Returns:
        retriever: A retriever instance for similarity search.

    Raises:
        QdrantOperationError: If retriever creation fails.
    """
    try:
        logger.info("Creating retriever from vector store")
        store = get_vector_store()
        retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        logger.info("Successfully created retriever")
        return retriever

    except (QdrantConnectionError, QdrantOperationError, EmbeddingModelError) as e:
        # Re-raise our custom exceptions
        logger.error(f"Failed to create retriever: {e}")
        raise

    except ValueError as e:
        logger.error(f"Invalid search parameters for retriever: {e}")
        raise QdrantOperationError(f"Invalid retriever search parameters: {e}")

    except AttributeError as e:
        logger.error(f"Vector store not properly initialized: {e}")
        raise QdrantOperationError(f"Vector store is not properly initialized: {e}")

    except Exception as e:
        logger.error(f"Unexpected error creating retriever: {type(e).__name__}: {e}")
        raise QdrantOperationError(f"Failed to create retriever: {e}")



