"""
Entity Extractor — Phase 1
LLM-based Named Entity Recognition using Groq LLaMA-3-8B.
Extracts typed entities from document chunks for KG construction.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from groq import Groq

logger = logging.getLogger(__name__)

ENTITY_TYPES = [
    "Person", "Team", "System", "Policy",
    "Incident", "Date", "Document", "CodeModule", "Metric"
]

EXTRACTION_PROMPT = """You are an expert Named Entity Recognition system for enterprise documents.

Extract ALL named entities from the following text. Return ONLY a valid JSON array.

Entity types to extract:
- Person: employees, authors, managers, stakeholders
- Team: departments, squads, groups, projects
- System: applications, services, databases, tools, platforms
- Policy: guidelines, procedures, standards, compliance rules
- Incident: outages, bugs, issues, failures, tickets
- Date: deadlines, release dates, events, quarters (e.g., Q3 2024)
- Document: reports, wikis, specs, manuals, tickets
- CodeModule: functions, classes, APIs, microservices, modules
- Metric: KPIs, SLAs, performance indicators, percentages, counts

Output format (JSON array only, no markdown, no explanation):
[
  {{"name": "entity name", "type": "EntityType", "context": "brief context from text"}},
  ...
]

If no entities found, return: []

Text to analyze:
{text}
"""


@dataclass
class ExtractedEntity:
    """A named entity extracted from a document chunk."""
    name: str
    entity_type: str
    context: str
    source_chunk_id: str
    source_doc_id: str
    properties: dict = field(default_factory=dict)


class EntityExtractor:
    """
    LLM-based Named Entity Recognition using Groq inference.
    Uses llama-3.1-8b-instant for cost-efficient bulk extraction.
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

    def extract_from_chunk(self, chunk, max_retries: int = 2) -> list[ExtractedEntity]:
        """
        Extract entities from a single DocumentChunk.

        Args:
            chunk: DocumentChunk object.

        Returns:
            List of ExtractedEntity objects.
        """
        for attempt in range(max_retries + 1):
            try:
                prompt = EXTRACTION_PROMPT.format(text=chunk.content[:3000])  # Limit token usage

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=1024,
                )

                raw = response.choices[0].message.content.strip()
                raw = self._clean_json_response(raw)
                entities_data = json.loads(raw)

                entities = []
                for item in entities_data:
                    if not isinstance(item, dict):
                        continue
                    entity_type = item.get("type", "").strip()
                    if entity_type not in ENTITY_TYPES:
                        continue
                    entities.append(ExtractedEntity(
                        name=item.get("name", "").strip(),
                        entity_type=entity_type,
                        context=item.get("context", "").strip(),
                        source_chunk_id=chunk.chunk_id,
                        source_doc_id=chunk.doc_id,
                    ))

                logger.debug(f"Extracted {len(entities)} entities from chunk {chunk.chunk_id[:8]}")
                return entities

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error for chunk {chunk.chunk_id[:8]} on attempt {attempt+1}: {e}")
                if attempt == max_retries:
                    return []
            except Exception as e:
                logger.error(f"Entity extraction failed for chunk {chunk.chunk_id[:8]}: {e}")
                return []
        
        return []

    def extract_from_chunks(self, chunks: list) -> list[ExtractedEntity]:
        """Extract entities from a list of chunks."""
        all_entities = []
        for i, chunk in enumerate(chunks):
            entities = self.extract_from_chunk(chunk)
            all_entities.extend(entities)
            if (i + 1) % 10 == 0:
                logger.info(f"Entity extraction progress: {i + 1}/{len(chunks)} chunks")

        logger.info(f"Total entities extracted: {len(all_entities)} from {len(chunks)} chunks")
        return all_entities

    def _clean_json_response(self, raw: str) -> str:
        """Strip markdown code blocks if LLM wrapped the JSON."""
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
        return raw.strip()
