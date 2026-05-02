"""
Embedder — Phase 1
Generates sentence embeddings using sentence-transformers.
Stores embedding_model_id and embedding_version with chunks for migration safety.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"
EMBEDDING_VERSION = "1.0"


class DocumentEmbedder:
    """
    Generates dense embeddings for document chunks using sentence-transformers.
    Tracks model ID and version for index migration safety.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._device = device
        self.model_id = model_name.split("/")[-1]
        self.model_version = EMBEDDING_VERSION

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self._device)
            logger.info(f"Embedding model loaded. Dimension: {self.get_dimension()}")

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        self._load_model()
        if not texts:
            return np.array([])

        logger.info(f"Embedding {len(texts)} texts in batches of {self.batch_size}")
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        self._load_model()
        return self._model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def embed_chunks(self, chunks: list) -> tuple[list, np.ndarray]:
        """
        Embed a list of DocumentChunks in place.
        Stamps embedding_model_id and embedding_version on each chunk.

        Args:
            chunks: List of DocumentChunk objects.

        Returns:
            Tuple of (updated_chunks, embeddings_array)
        """
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding_model_id = self.model_id
            chunk.embedding_version = self.model_version

        logger.info(f"Embedded {len(chunks)} chunks with model '{self.model_id}'")
        return chunks, embeddings
