"""TF-IDF based retriever over policy chunks (zero external dependencies
beyond scikit-learn; no model download required, works fully offline)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .policy_store import PolicyChunk


@dataclass
class RetrievedChunk:
    chunk: PolicyChunk
    score: float


class TfidfRetriever:
    """Simple, dependable sparse retriever.

    Swap this out for a dense retriever (sentence-transformers / OpenAI
    embeddings / a vector DB such as Chroma or Weaviate) in production --
    the interface (`retrieve`) stays the same.
    """

    def __init__(self, chunks: Sequence[PolicyChunk]):
        if not chunks:
            raise ValueError("Cannot build a retriever over zero chunks.")
        self.chunks: List[PolicyChunk] = list(chunks)
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        corpus = [f"{c.title} {c.heading} {c.text}" for c in self.chunks]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        procedure_category: str | None = None,
    ) -> List[RetrievedChunk]:
        """Returns the top_k most relevant chunks for `query`.

        If `procedure_category` is given, retrieval is restricted to chunks
        belonging to that category first (mirrors how a real system would
        route a claim to the correct policy family before dense retrieval).
        """
        query_vec = self._vectorizer.transform([query])
        scores = linear_kernel(query_vec, self._matrix).ravel()

        candidates = list(enumerate(scores))
        if procedure_category is not None:
            filtered = [
                (i, s)
                for i, s in candidates
                if self.chunks[i].procedure_category == procedure_category
            ]
            if filtered:
                candidates = filtered

        candidates.sort(key=lambda pair: pair[1], reverse=True)
        top = candidates[:top_k]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(s)) for i, s in top]
