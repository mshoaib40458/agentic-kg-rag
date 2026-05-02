"""
Test Suite — Ingestion Pipeline
Tests: DocumentParser, DocumentChunker, DocumentEmbedder
"""
import os
import pytest
import tempfile
from pathlib import Path

from src.ingestion.document_parser import DocumentParser, ParsedDocument
from src.ingestion.chunker import DocumentChunker, DocumentChunk


class TestDocumentParser:
    def setup_method(self):
        self.parser = DocumentParser()

    def test_parse_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world. This is a test document.\nSecond line.")
        doc = self.parser.parse(str(f))
        assert isinstance(doc, ParsedDocument)
        assert "Hello world" in doc.content
        assert doc.file_type == ".txt"
        assert doc.doc_id

    def test_parse_markdown(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\n\nSome content here.\n\n## Section 2\n\nMore text.")
        doc = self.parser.parse(str(f))
        assert "Title" in doc.content
        assert doc.file_type == ".md"

    def test_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("data")
        with pytest.raises(ValueError, match="Unsupported file type"):
            self.parser.parse(str(f))

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            self.parser.parse("/nonexistent/file.txt")

    def test_metadata_preserved(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Test content.")
        doc = self.parser.parse(str(f), metadata={"department": "engineering"})
        assert doc.metadata.get("department") == "engineering"
        assert doc.metadata.get("filename") == "test.txt"


class TestDocumentChunker:
    def setup_method(self):
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)

    def _make_doc(self, content: str) -> ParsedDocument:
        return ParsedDocument(
            doc_id="test-doc-id",
            filename="test.txt",
            file_type=".txt",
            content=content,
            metadata={"filename": "test.txt"},
        )

    def test_chunk_short_doc(self):
        doc = self._make_doc("Short document content.")
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) >= 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunk_long_doc(self):
        content = "This is a sentence. " * 200
        doc = self._make_doc(content)
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) > 1

    def test_chunk_traceability(self):
        doc = self._make_doc("Content for testing chunk traceability." * 50)
        chunks = self.chunker.chunk_document(doc)
        for chunk in chunks:
            assert chunk.doc_id == "test-doc-id"
            assert chunk.filename == "test.txt"
            assert chunk.chunk_id
            assert chunk.total_chunks == len(chunks)

    def test_empty_doc_returns_empty(self):
        doc = self._make_doc("")
        chunks = self.chunker.chunk_document(doc)
        assert chunks == []

    def test_rbac_roles_attached(self):
        doc = self._make_doc("Test content for RBAC." * 20)
        chunks = self.chunker.chunk_document(doc, access_roles=["admin"])
        for chunk in chunks:
            assert "admin" in chunk.access_roles
