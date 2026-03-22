# Vector database
import os
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_huggingface import HuggingFaceEmbeddings

_MODEL_NAME = "sdadas/mmlw-retrieval-roberta-large-v2"
_QUERY_PREFIX = "[query]: "


class QueryPrefixEmbeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings that prepends a prefix to queries for asymmetric retrieval."""

    query_prefix: str = ""

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(self.query_prefix + text)


# Support both Docker and local setups
_qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
_vector_database_client = QdrantClient(url=_qdrant_url)

# vector_size = len(process_data.embeddings[0]['embedding'])


class QdrantVectorDatabase:
    """
    A class to encapsulate Qdrant vector database operations.
    """

    def __init__(self, url: str | None = None):
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=self.url)
        self.embeddings = QueryPrefixEmbeddings(
            model_name=_MODEL_NAME,
            query_prefix=_QUERY_PREFIX,
            model_kwargs={"model_kwargs": {"torch_dtype": "float16"}},
        )

    def delete_collection_if_exists(self, collection_name: str):
        """
        Delete a collection in Qdrant if it exists.
        Args:
            vector_database_client (QdrantClient): An instance of QdrantClient.
            collection_name (str): The name of the collection to delete.
        """
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            print(f"'{collection_name}' DEL")

    def create_collection_if_not_exists(self, collection_name: str, vector_size: int):
        """
        Create a collection in Qdrant if it does not exist.
        Args:
            vector_database_client (QdrantClient): An instance of QdrantClient.
            collection_name (str): The name of the collection to create.
            vector_size (int): The size of the vectors to be stored in the collection.
        """
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def collection_has_data(self, collection_name: str) -> bool:
        """Check if collection exists and has points."""
        if not self.client.collection_exists(collection_name):
            return False
        info = self.client.get_collection(collection_name)
        return info.points_count > 0

    def collection_vector_size_matches(
        self, collection_name: str, expected_size: int
    ) -> bool:
        """Check if existing collection's vector dimension matches expected size."""
        if not self.client.collection_exists(collection_name):
            return True
        info = self.client.get_collection(collection_name)
        return info.config.params.vectors.size == expected_size

    @staticmethod
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
            payload = {"text": emb["text"], **emb["metadata"]}
            points.append(
                {
                    "id": i,
                    "vector": emb["embedding"].tolist(),
                    "payload": payload,
                }
            )
        return points

    def upload_to_qdrant(self, collection_name: str, points: list):
        """
        Upload embeddings to a Qdrant collection.
        Args:
            vector_database_client (QdrantClient): An instance of QdrantClient.
            collection_name (str): The name of the collection to upload to.
            points (list): A list of dictionaries containing embeddings and metadata.
        """

        self.client.upsert(collection_name=collection_name, points=points)
        print("✅ Upload complete")

    # lch_vector_store = QdrantVectorStore(client=_vector_database_client,
    #                                     collection_name=_collection_name,
    #                                     embedding=embeddings_for_qdrant_vector_store,
    #                                     content_payload_key="text",)

    def get_vector_store(self, collection_name: str) -> QdrantVectorStore:
        """
        Create QdrantVectorStore lazily (only after collection exists).
        """
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
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

    def create_retriever(self, collection_name: str):
        """
        Create a retriever after data is loaded.
        """
        store = self.get_vector_store(collection_name)
        return store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )
