"""
KG Builder — Phase 1
Writes extracted entities and relationships to Neo4j.
Uses MERGE to avoid duplicates. Attaches source traceability to all triples.
"""

import logging
import os
from typing import Optional

from neo4j import GraphDatabase, Driver

from src.ingestion.entity_extractor import ExtractedEntity
from src.ingestion.relation_extractor import ExtractedRelationship

logger = logging.getLogger(__name__)

# Neo4j schema constraints setup queries
CONSTRAINT_QUERIES = [
    "CREATE CONSTRAINT entity_name_type IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
    "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
]


class KGBuilder:
    """
    Writes entities and relationships to Neo4j Knowledge Graph.
    Enforces schema via constraints and indexes.
    Uses MERGE for idempotent writes.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.database = database
        self._driver: Optional[Driver] = None

    def connect(self):
        """Establish Neo4j driver connection."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )
            self._driver.verify_connectivity()
            logger.info(f"✓ Connected to Neo4j at {self.uri}")

    def disconnect(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    def setup_schema(self):
        """Create constraints and indexes if they don't exist."""
        self.connect()
        with self._driver.session(database=self.database) as session:
            for query in CONSTRAINT_QUERIES:
                try:
                    session.run(query)
                except Exception as e:
                    logger.warning(f"Schema setup warning: {e}")
        logger.info("✓ Neo4j schema constraints and indexes applied")

    def write_entities(self, entities: list[ExtractedEntity]) -> int:
        """
        Write entities to Neo4j using MERGE (idempotent).

        Args:
            entities: List of ExtractedEntity objects.

        Returns:
            Number of entities written.
        """
        self.connect()
        count = 0

        with self._driver.session(database=self.database) as session:
            for entity in entities:
                if not entity.name or not entity.entity_type:
                    continue
                try:
                    session.run(
                        """
                        MERGE (e:Entity {name: $name, type: $type})
                        ON CREATE SET
                            e.created_at = timestamp(),
                            e.source_doc_ids = [$doc_id],
                            e.source_chunk_ids = [$chunk_id],
                            e.context = $context,
                            e.labels = [$type]
                        ON MATCH SET
                            e.source_doc_ids = CASE
                                WHEN NOT $doc_id IN e.source_doc_ids
                                THEN e.source_doc_ids + [$doc_id]
                                ELSE e.source_doc_ids
                            END,
                            e.source_chunk_ids = CASE
                                WHEN NOT $chunk_id IN e.source_chunk_ids
                                THEN e.source_chunk_ids + [$chunk_id]
                                ELSE e.source_chunk_ids
                            END,
                            e.updated_at = timestamp(),
                            e.labels = CASE
                                WHEN NOT $type IN e.labels
                                THEN e.labels + [$type]
                                ELSE e.labels
                            END
                        """,
                        name=entity.name,
                        type=entity.entity_type,
                        doc_id=entity.source_doc_id,
                        chunk_id=entity.source_chunk_id,
                        context=entity.context,
                    )
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to write entity '{entity.name}': {e}")

        logger.info(f"✓ Wrote {count} entities to Neo4j")
        return count

    def write_relationships(self, relationships: list[ExtractedRelationship]) -> int:
        """
        Write relationships to Neo4j using MERGE (idempotent) with injection protection.

        Args:
            relationships: List of ExtractedRelationship objects.

        Returns:
            Number of relationships written.
        """
        # Allowed relationship types from relation_extractor
        ALLOWED_REL_TYPES = {
            "OWNS", "DEPENDS_ON", "CAUSED_BY", "VIOLATES",
            "PART_OF", "MENTIONS", "IMPLEMENTS", "REPORTS_TO"
        }
        
        self.connect()
        count = 0

        with self._driver.session(database=self.database) as session:
            for rel in relationships:
                if not rel.source_name or not rel.target_name or not rel.relationship:
                    continue
                
                # SECURITY FIX: Validate relationship type against whitelist
                rel_type = rel.relationship.upper()
                if rel_type not in ALLOWED_REL_TYPES:
                    logger.warning(f"Skipping unknown relationship type: {rel_type}")
                    continue
                
                try:
                    query = f"""
                        MERGE (source:Entity {{name: $source_name}})
                        ON CREATE SET source.type = $source_type
                        MERGE (target:Entity {{name: $target_name}})
                        ON CREATE SET target.type = $target_type
                        MERGE (source)-[r:{rel_type}]->(target)
                        ON CREATE SET
                            r.confidence = $confidence,
                            r.source_doc_id = $doc_id,
                            r.source_chunk_id = $chunk_id,
                            r.created_at = timestamp()
                        ON MATCH SET
                            r.confidence = CASE
                                WHEN $confidence > r.confidence THEN $confidence
                                ELSE r.confidence
                            END,
                            r.updated_at = timestamp()
                    """
                    session.run(
                        query,
                        source_name=rel.source_name,
                        source_type=rel.source_type,
                        target_name=rel.target_name,
                        target_type=rel.target_type,
                        confidence=rel.confidence,
                        doc_id=rel.source_doc_id,
                        chunk_id=rel.source_chunk_id,
                    )
                    count += 1
                except Exception as e:
                    logger.error(
                        f"Failed to write relationship "
                        f"'{rel.source_name}' -{rel.relationship}-> '{rel.target_name}': {e}"
                    )

        logger.info(f"✓ Wrote {count} relationships to Neo4j")
        return count

    def get_stats(self) -> dict:
        """Return basic graph statistics using separate count queries to avoid Cartesian product."""
        self.connect()
        try:
            with self._driver.session(database=self.database) as session:
                entity_result = session.run("MATCH (n:Entity) RETURN count(n) AS entity_count")
                entity_count = entity_result.single()["entity_count"]

                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) AS rel_count")
                rel_count = rel_result.single()["rel_count"]

                return {
                    "entity_count": entity_count,
                    "relationship_count": rel_count,
                }
        except Exception as e:
            logger.warning(f"Failed to fetch graph stats: {e}")
            return {"entity_count": 0, "relationship_count": 0}

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
