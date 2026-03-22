import torch.nn
from sentence_transformers import CrossEncoder

_RERANKER_MODEL_NAME = "sdadas/polish-reranker-roberta-v3"
_RERANKER_MAX_LENGTH = 8192
_RERANKER_TOP_K = 3


class Reranker:
    """Reranker using sdadas/polish-reranker-roberta-v3 CrossEncoder for Polish text."""

    def __init__(
        self,
        model_name: str = _RERANKER_MODEL_NAME,
        max_length: int = _RERANKER_MAX_LENGTH,
        top_k: int = _RERANKER_TOP_K,
    ):
        self.top_k = top_k
        self.model = CrossEncoder(
            model_name,
            activation_fn=torch.nn.Identity(),
            max_length=max_length,
        )

    def rerank(self, query: str, documents: list) -> list:
        """
        Rerank documents by relevance to the query using a cross-encoder.
        Args:
            query: The user's search query.
            documents: List of langchain Document objects from retriever.
        Returns:
            List of top_k Document objects sorted by relevance score (descending).
        """
        if not documents:
            return documents

        top_docs, _ = self._rerank_with_scores(query, documents)
        return top_docs

    def rerank_with_debug(self, query: str, documents: list) -> tuple:
        """
        Rerank and return debug information.
        Returns:
            (top_k_docs, all_scored_pairs) where all_scored_pairs is
            [(score, doc), ...] sorted by score descending.
        """
        if not documents:
            return documents, []

        return self._rerank_with_scores(query, documents)

    def _rerank_with_scores(self, query: str, documents: list) -> tuple:
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = sorted(
            zip(scores, documents), key=lambda x: x[0], reverse=True
        )
        top_docs = [doc for _, doc in scored_docs[: self.top_k]]
        all_scored = [(float(s), doc) for s, doc in scored_docs]
        return top_docs, all_scored
