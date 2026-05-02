"""
Relation Extractor — Phase 1
LLM-based Relationship Extraction using Groq LLaMA-3-8B.
Extracts typed relationships between entities for KG construction.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from groq import Groq
from src.ingestion.entity_extractor import ExtractedEntity

logger = logging.getLogger(__name__)

RELATIONSHIP_TYPES = [
    "owns", "depends_on", "caused_by", "violates",
    "part_of", "mentions", "implements", "reports_to"
]

RELATION_PROMPT = """You are an expert Relationship Extraction system for enterprise knowledge graphs.

Given a list of named entities and the source text, identify relationships between entities.

Valid relationship types:
- owns: Person or Team → System or Policy (e.g., "Alice owns the Auth Service")
- depends_on: System → System (e.g., "PaymentAPI depends_on DatabaseService")
- caused_by: Incident → System or Person (e.g., "Outage caused_by deploy by Bob")
- violates: Incident → Policy (e.g., "INC-123 violates procurement policy")
- part_of: Person → Team, or Module → System (e.g., "Alice part_of Backend Team")
- mentions: Document → any Entity
- implements: CodeModule → Policy (e.g., "AuthModule implements GDPR policy")
- reports_to: Person → Person (e.g., "Alice reports_to Bob")

Return ONLY a valid JSON array of relationships. No markdown, no explanation.
Format:
[
  {{
    "source": "source entity name",
    "source_type": "EntityType",
    "relationship": "relationship_type",
    "target": "target entity name",
    "target_type": "EntityType",
    "confidence": 0.9
  }},
  ...
]

If no relationships found, return: []

Entities:
{entities}

Source text:
{text}
"""


@dataclass
class ExtractedRelationship:
    """A typed relationship between two entities."""
    source_name: str
    source_type: str
    relationship: str
    target_name: str
    target_type: str
    confidence: float
    source_chunk_id: str
    source_doc_id: str


class RelationExtractor:
    """
    LLM-based Relationship Extraction using Groq inference.
    Extracts typed relationships between named entities.
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.model = model
        self.temperature = temperature
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))

    def extract_from_chunk(
        self,
        chunk,
        entities: list[ExtractedEntity],
        max_retries: int = 2
    ) -> list[ExtractedRelationship]:
        """
        Extract relationships between entities in a single chunk.

        Args:
            chunk: DocumentChunk object.
            entities: Entities already extracted from this chunk.

        Returns:
            List of ExtractedRelationship objects.
        """
        if not entities:
            return []

        for attempt in range(max_retries + 1):
            try:
                entity_list = [
                    {"name": e.name, "type": e.entity_type} for e in entities
                ]
                prompt = RELATION_PROMPT.format(
                    entities=json.dumps(entity_list, indent=2),
                    text=chunk.content[:3000],
                )

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=1024,
                )

                raw = response.choices[0].message.content.strip()
                raw = self._clean_json_response(raw)
                relations_data = json.loads(raw)

                relationships = []
                for item in relations_data:
                    if not isinstance(item, dict):
                        continue
                    rel_type = item.get("relationship", "").strip()
                    if rel_type not in RELATIONSHIP_TYPES:
                        continue
                    relationships.append(ExtractedRelationship(
                        source_name=item.get("source", "").strip(),
                        source_type=item.get("source_type", "").strip(),
                        relationship=rel_type,
                        target_name=item.get("target", "").strip(),
                        target_type=item.get("target_type", "").strip(),
                        confidence=float(item.get("confidence", 0.8)),
                        source_chunk_id=chunk.chunk_id,
                        source_doc_id=chunk.doc_id,
                    ))

                logger.debug(f"Extracted {len(relationships)} relationships from chunk {chunk.chunk_id[:8]}")
                return relationships

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error for chunk {chunk.chunk_id[:8]} on attempt {attempt+1}: {e}")
                if attempt == max_retries:
                    return []
            except Exception as e:
                logger.error(f"Relation extraction failed for chunk {chunk.chunk_id[:8]}: {e}")
                return []
        
        return []

    def extract_from_chunks(
        self,
        chunks: list,
        entities_by_chunk: dict[str, list[ExtractedEntity]],
    ) -> list[ExtractedRelationship]:
        """Extract relationships across all chunks."""
        all_relationships = []
        for i, chunk in enumerate(chunks):
            entities = entities_by_chunk.get(chunk.chunk_id, [])
            relationships = self.extract_from_chunk(chunk, entities)
            all_relationships.extend(relationships)
            if (i + 1) % 10 == 0:
                logger.info(f"Relation extraction: {i + 1}/{len(chunks)} chunks")

        logger.info(f"Total relationships extracted: {len(all_relationships)}")
        return all_relationships

    def _clean_json_response(self, raw: str) -> str:
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
        return raw.strip()
