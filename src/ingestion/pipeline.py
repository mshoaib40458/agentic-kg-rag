"""
Ingestion Pipeline — Phase 1
Orchestrates the full document-to-knowledge pipeline:
Parse → Chunk → Embed → Store FAISS → NER → RE → Store KG
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.ingestion.chunker import DocumentChunk, DocumentChunker
from src.ingestion.document_parser import DocumentParser, ParsedDocument
from src.ingestion.embedder import DocumentEmbedder
from src.ingestion.entity_extractor import EntityExtractor, ExtractedEntity
from src.ingestion.kg_builder import KGBuilder
from src.ingestion.relation_extractor import ExtractedRelationship, RelationExtractor

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Summary of a completed ingestion run."""
    total_documents: int
    total_chunks: int
    total_entities: int
    total_relationships: int
    failed_documents: list[str]
    duration_seconds: float


class IngestionPipeline:
    """
    Full document ingestion pipeline.
    Parses, chunks, embeds, extracts entities/relations,
    stores in FAISS vector store and Neo4j knowledge graph.
    """

    def __init__(
        self,
        vector_store=None,       # Injected VectorStore instance
        embedder=None,           # Injected DocumentEmbedder singleton (prevents reload per request)
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant",
        groq_api_key: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        run_kg_extraction: bool = True,
    ):
        self.vector_store = vector_store
        self.run_kg_extraction = run_kg_extraction

        # Initialize pipeline components
        self.parser = DocumentParser()
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # Prefer injected singleton to avoid reloading the model on every request
        self.embedder = embedder or DocumentEmbedder(model_name=embedding_model)

        self.entity_extractor = EntityExtractor(
            model=llm_model,
            api_key=groq_api_key or os.getenv("GROQ_API_KEY"),
        )
        self.relation_extractor = RelationExtractor(
            model=llm_model,
            api_key=groq_api_key or os.getenv("GROQ_API_KEY"),
        )
        self.kg_builder = KGBuilder(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
        )

    def ingest_file(
        self,
        file_path: str,
        metadata: Optional[dict] = None,
        access_roles: Optional[list[str]] = None,
    ) -> IngestionResult:
        """
        Ingest a single document file through the full pipeline.

        Args:
            file_path: Path to the document file.
            metadata: Optional metadata to attach to all chunks.
            access_roles: RBAC roles allowed to access this document.

        Returns:
            IngestionResult summary.
        """
        import time
        start_time = time.time()
        failed_documents = []

        try:
            logger.info(f"▶ Ingesting file: {file_path}")

            # Step 1: Parse
            document = self.parser.parse(file_path, metadata)
            logger.info(f"  [1/5] Parsed: {document.filename} ({len(document.content)} chars)")

            # Step 2: Chunk
            chunks = self.chunker.chunk_document(document, access_roles)
            logger.info(f"  [2/5] Chunked: {len(chunks)} chunks")

            # Step 3: Embed + Store to Vector Store (defer save until all chunks added)
            vector_chunks_saved = False
            if self.vector_store and chunks:
                try:
                    chunks, embeddings = self.embedder.embed_chunks(chunks)
                    self.vector_store.add_no_save(chunks, embeddings)
                    vector_chunks_saved = True
                    logger.info(f"  [3/5] Embedded and stored {len(chunks)} chunks to FAISS")
                except Exception as e:
                    logger.error(f"  [3/5] FAISS embedding/storage failed: {e}")
                    raise RuntimeError(f"Failed to embed and store chunks: {e}")
            else:
                logger.warning("  [3/5] No vector store injected — skipping embedding storage")

            total_entities = 0
            total_relationships = 0

            # Step 4 & 5: NER + RE + KG Storage
            if self.run_kg_extraction and chunks:
                try:
                    logger.info(f"  [4/5] Running NER on {len(chunks)} chunks...")
                    entities = self.entity_extractor.extract_from_chunks(chunks)

                    # Group entities by chunk_id for relation extraction
                    entities_by_chunk = {}
                    for ent in entities:
                        entities_by_chunk.setdefault(ent.source_chunk_id, []).append(ent)
                    
                    logger.info(f"  [5/5] Running RE and building KG...")
                    relationships = self.relation_extractor.extract_from_chunks(chunks, entities_by_chunk)

                    # Setup schema once then write
                    self.kg_builder.setup_schema()
                    total_entities = self.kg_builder.write_entities(entities)
                    total_relationships = self.kg_builder.write_relationships(relationships)
                except Exception as e:
                    logger.error(f"  [4-5/5] KG extraction failed: {e}")
                    if vector_chunks_saved:
                        logger.warning("  [⚠] Continuing ingestion without KG — document will be vector-searchable only")
                    else:
                        raise
            else:
                logger.info("  [4-5/5] KG extraction skipped (run_kg_extraction=False)")

            # Save FAISS index once (batch save — avoids O(n) incremental writes)
            if self.vector_store and chunks and vector_chunks_saved:
                try:
                    self.vector_store.save()
                    logger.info("  [✓] FAISS index saved to disk")
                except Exception as e:
                    logger.error(f"  [✗] FAISS save failed: {e} — ingestion incomplete")
                    failed_documents.append(file_path)
                    raise RuntimeError(f"Failed to persist vector store: {e}")

            duration = time.time() - start_time
            logger.info(f"✓ Ingestion complete in {duration:.1f}s | "
                       f"chunks={len(chunks)} entities={total_entities} rels={total_relationships}")

            return IngestionResult(
                total_documents=1,
                total_chunks=len(chunks),
                total_entities=total_entities,
                total_relationships=total_relationships,
                failed_documents=failed_documents,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"✗ Ingestion failed for {file_path}: {e}")
            failed_documents.append(file_path)
            return IngestionResult(
                total_documents=1,
                total_chunks=0,
                total_entities=0,
                total_relationships=0,
                failed_documents=failed_documents,
                duration_seconds=0.0,
            )

    def ingest_directory(
        self,
        dir_path: str,
        metadata: Optional[dict] = None,
        access_roles: Optional[list[str]] = None,
    ) -> IngestionResult:
        """Ingest all supported documents from a directory."""
        import time
        start_time = time.time()

        dir_path = Path(dir_path)
        supported_exts = DocumentParser.SUPPORTED_EXTENSIONS
        files = [f for f in dir_path.rglob("*") if f.suffix.lower() in supported_exts]

        logger.info(f"Found {len(files)} documents to ingest from {dir_path}")

        totals = IngestionResult(0, 0, 0, 0, [], 0.0)

        for file_path in files:
            result = self.ingest_file(str(file_path), metadata, access_roles)
            totals.total_documents += result.total_documents
            totals.total_chunks += result.total_chunks
            totals.total_entities += result.total_entities
            totals.total_relationships += result.total_relationships
            totals.failed_documents.extend(result.failed_documents)

        totals.duration_seconds = time.time() - start_time
        logger.info(
            f"✓ Batch ingestion complete: "
            f"{totals.total_documents} docs, {totals.total_chunks} chunks, "
            f"{totals.total_entities} entities, {totals.total_relationships} rels | "
            f"{totals.duration_seconds:.1f}s"
        )
        return totals
