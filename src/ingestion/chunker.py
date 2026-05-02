"""
Chunker — Phase 1
Splits ParsedDocuments into overlapping chunks using RecursiveCharacterTextSplitter.
Preserves full metadata + traceability per chunk.
"""

import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Optional
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.ingestion.document_parser import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single text chunk with full traceability to its source document."""
    chunk_id: str
    doc_id: str
    filename: str
    content: str
    chunk_index: int
    total_chunks: int
    char_start: int
    char_end: int
    embedding_model_id: str = ""       # set by embedder
    embedding_version: str = ""        # set by embedder
    access_roles: list[str] = field(default_factory=lambda: ["admin", "user", "auditor"])
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())


class DocumentChunker:
    """
    Splits documents into overlapping chunks with configurable size/overlap.
    Uses RecursiveCharacterTextSplitter for intelligent boundary detection.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        length_function=len,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_document(
        self,
        document: ParsedDocument,
        access_roles: Optional[list[str]] = None,
    ) -> list[DocumentChunk]:
        """
        Chunk a single ParsedDocument.

        Args:
            document: Parsed document to chunk.
            access_roles: RBAC roles that can access these chunks.

        Returns:
            List of DocumentChunk objects.
        """
        if not document.content.strip():
            logger.warning(f"Document '{document.filename}' has empty content — skipping.")
            return []

        # Generate chunks with character positions
        raw_chunks = self.splitter.create_documents(
            texts=[document.content],
            metadatas=[document.metadata],
        )

        chunks = []
        char_offset = 0

        for idx, raw_chunk in enumerate(raw_chunks):
            text = raw_chunk.page_content
            char_start = document.content.find(text, char_offset)
            char_end = char_start + len(text) if char_start != -1 else char_offset + len(text)
            char_offset = max(0, char_end - self.chunk_overlap)

            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=document.doc_id,
                filename=document.filename,
                content=text,
                chunk_index=idx,
                total_chunks=len(raw_chunks),
                char_start=max(0, char_start),
                char_end=char_end,
                access_roles=access_roles or ["admin", "user", "auditor"],
                metadata={
                    **document.metadata,
                    "chunk_index": idx,
                    "total_chunks": len(raw_chunks),
                    "char_start": max(0, char_start),
                    "char_end": char_end,
                },
            )
            chunks.append(chunk)

        logger.info(
            f"Chunked '{document.filename}' → {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def chunk_documents(
        self,
        documents: list[ParsedDocument],
        access_roles: Optional[list[str]] = None,
    ) -> list[DocumentChunk]:
        """Chunk a list of documents."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc, access_roles)
            all_chunks.extend(chunks)
        logger.info(f"Total chunks generated: {len(all_chunks)} from {len(documents)} documents")
        return all_chunks
