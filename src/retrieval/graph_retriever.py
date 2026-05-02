"""
Graph Retriever — Phase 3
Natural language → Cypher via Groq LLaMA-3-70B.
Multi-hop graph traversal with schema-constrained, RBAC-enforced queries.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from groq import Groq
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

# ── Schema Registry (whitelisted types only) ────────────────
ALLOWED_NODE_LABELS = [
    "Entity", "Person", "Team", "System", "Policy",
    "Incident", "Date", "Document", "CodeModule", "Metric"
]
ALLOWED_REL_TYPES = [
    "OWNS", "DEPENDS_ON", "CAUSED_BY", "VIOLATES",
    "PART_OF", "MENTIONS", "IMPLEMENTS", "REPORTS_TO"
]

CYPHER_GEN_PROMPT = """You are an expert Neo4j Cypher query generator for an enterprise knowledge graph.

Graph Schema:
- Node label: Entity (with properties: name, type, context, source_doc_ids)
- Entity types: {entity_types}
- Relationship types: {rel_types}
- All relationships: (source:Entity)-[r:REL_TYPE]->(target:Entity)

STRICT RULES:
1. ONLY use node label: Entity
2. ONLY use relationship types from the allowed list above
3. ALWAYS include LIMIT {limit} at the end
4. NEVER use MATCH (n) without a WHERE clause
5. Use case-insensitive matching: WHERE toLower(e.name) CONTAINS toLower('...')
6. Return: entity names, types, relationship types, and source_doc_ids
7. Return ONLY the Cypher query, no explanation, no markdown

Natural language query: {query}

Generate Cypher query:"""



@dataclass
class GraphPath:
    """A single traversal path in the knowledge graph."""
    nodes: list[dict]
    relationships: list[dict]
    path_string: str
    source_doc_ids: list[str] = field(default_factory=list)


@dataclass
class GraphSearchResult:
    """Result from a knowledge graph query."""
    query: str
    cypher: str
    paths: list[GraphPath]
    entities: list[dict]
    raw_records: list[dict]


class GraphRetriever:
    """
    Retrieves information from the Neo4j knowledge graph.
    Converts natural language to Cypher with schema constraints and RBAC injection.
    Supports multi-hop traversal up to configurable depth.
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        database: str = "neo4j",
        groq_api_key: Optional[str] = None,
        llm_model: str = "llama-3.3-70b-versatile",
        hop_depth: int = 3,
        cypher_limit: int = 50,
    ):
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_username = neo4j_username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "")
        self.database = database
        self.hop_depth = hop_depth
        self.cypher_limit = cypher_limit
        self.llm_model = llm_model

        self._driver = None
        self.groq_client = Groq(api_key=groq_api_key or os.getenv("GROQ_API_KEY"))

    def _get_driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_username, self.neo4j_password),
            )
        return self._driver

    def close(self):
        """Explicitly close Neo4j driver connection. Call on app shutdown."""
        if self._driver:
            try:
                self._driver.close()
                self._driver = None
                logger.info("✓ Neo4j driver closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j driver: {e}")

    def verify_connectivity(self) -> bool:
        """Verify Neo4j connection is alive. Call during app startup."""
        try:
            driver = self._get_driver()
            driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Neo4j connectivity check failed: {e}")
            return False

    # ── Cypher Generation ──────────────────────────────────────

    def generate_cypher(self, nl_query: str) -> str:
        """
        Convert a natural language query to a schema-constrained Cypher query.

        Args:
            nl_query: Natural language query string.

        Returns:
            Valid Cypher query string.
        """
        prompt = CYPHER_GEN_PROMPT.format(
            entity_types=", ".join(ALLOWED_NODE_LABELS),
            rel_types=", ".join(ALLOWED_REL_TYPES),
            limit=self.cypher_limit,
            query=nl_query,
        )

        response = self.groq_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        cypher = response.choices[0].message.content.strip()
        cypher = self._clean_cypher(cypher)
        cypher = self._enforce_safety(cypher)

        logger.debug(f"Generated Cypher: {cypher}")
        return cypher

    def _clean_cypher(self, cypher: str) -> str:
        """Strip markdown code fences if present."""
        cypher = cypher.strip()
        if cypher.startswith("```"):
            lines = cypher.split("\n")
            cypher = "\n".join(lines[1:-1])
        return cypher.strip()

    def _enforce_safety(self, cypher: str) -> str:
        """Enforce safety: ensure LIMIT exists, block dangerous patterns."""
        # Ensure LIMIT clause
        if "LIMIT" not in cypher.upper():
            cypher = cypher.rstrip(";") + f" LIMIT {self.cypher_limit}"
        else:
            # Replace existing LIMIT with safe limit
            cypher = re.sub(r'LIMIT\s+\d+', f'LIMIT {self.cypher_limit}', cypher, flags=re.IGNORECASE)

        # Block dangerous statements
        dangerous = ["DELETE", "DETACH", "REMOVE", "SET", "CREATE", "MERGE", "DROP"]
        for keyword in dangerous:
            if re.search(rf"\b{keyword}\b", cypher, re.IGNORECASE):
                logger.warning(f"Blocked dangerous Cypher keyword: {keyword}")
                return "MATCH (n:Entity) RETURN n.name, n.type LIMIT 10"

        return cypher

    # ── Query Execution ────────────────────────────────────────

    def query(
        self,
        nl_query: str,
        user_role: str = "user",
        hop_depth: Optional[int] = None,
    ) -> GraphSearchResult:
        """
        Execute a natural language query against the knowledge graph.

        Args:
            nl_query: Natural language query.
            user_role: User's RBAC role.
            hop_depth: Override default hop depth.

        Returns:
            GraphSearchResult with paths and entities.
        """
        cypher = self.generate_cypher(nl_query)

        # Inject RBAC role filter: append a WHERE/AND clause for role gating
        # This ensures non-admin roles cannot read restricted entities
        cypher = self._inject_rbac(cypher, user_role)

        try:
            driver = self._get_driver()
            with driver.session(database=self.database) as session:
                result = session.run(cypher)
                records = [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Cypher execution failed: {e}\nQuery: {cypher}")
            # Fallback: simple entity name search
            return self._fallback_search(nl_query, user_role)

        entities = self._extract_entities(records)
        paths = self._extract_paths(records)

        return GraphSearchResult(
            query=nl_query,
            cypher=cypher,
            paths=paths,
            entities=entities,
            raw_records=records,
        )


    def multihop_query(
        self,
        start_entity: str,
        end_entity: Optional[str] = None,
        hop_depth: Optional[int] = None,
        user_role: str = "user",
    ) -> GraphSearchResult:
        """
        Execute a multi-hop traversal query between entities.
        Uses parameterized queries to prevent Cypher injection.

        Args:
            start_entity: Starting entity name.
            end_entity: Optional target entity name.
            hop_depth: Max hop depth (default: self.hop_depth).
            user_role: User's RBAC role.
        """
        depth = hop_depth or self.hop_depth
        
        rbac_condition = ""
        if user_role != "admin":
            rbac_condition = f"AND all(n IN nodes(path) WHERE '{user_role}' IN n.access_roles OR n.access_roles IS NULL OR size(n.access_roles) = 0)"

        if end_entity:
            cypher = f"""
                MATCH path = (start:Entity)-[*1..{depth}]-(end:Entity)
                WHERE toLower(start.name) CONTAINS toLower($start_entity)
                AND toLower(end.name) CONTAINS toLower($end_entity)
                {rbac_condition}
                RETURN path,
                       [n IN nodes(path) | {{name: n.name, type: n.type}}] as node_list,
                       [r IN relationships(path) | type(r)] as rel_list
                LIMIT {self.cypher_limit}
            """
            params = {"start_entity": start_entity, "end_entity": end_entity}
        else:
            cypher = f"""
                MATCH path = (start:Entity)-[*1..{depth}]-(neighbor:Entity)
                WHERE toLower(start.name) CONTAINS toLower($start_entity)
                {rbac_condition}
                RETURN path,
                       [n IN nodes(path) | {{name: n.name, type: n.type}}] as node_list,
                       [r IN relationships(path) | type(r)] as rel_list
                LIMIT {self.cypher_limit}
            """
            params = {"start_entity": start_entity}

        try:
            driver = self._get_driver()
            with driver.session(database=self.database) as session:
                result = session.run(cypher, **params)
                records = [dict(record) for record in result]

            paths = []
            for record in records:
                node_list = record.get("node_list", [])
                rel_list = record.get("rel_list", [])
                path_str = " → ".join(
                    f"{n.get('name', '?')} [{n.get('type', '?')}]" for n in node_list
                )
                paths.append(GraphPath(
                    nodes=node_list,
                    relationships=[{"type": r} for r in rel_list],
                    path_string=path_str,
                ))

            return GraphSearchResult(
                query=f"multihop: {start_entity} → {end_entity or '...'}",
                cypher=cypher,
                paths=paths,
                entities=[n for r in paths for n in r.nodes],
                raw_records=records,
            )

        except Exception as e:
            logger.error(f"Multi-hop query failed: {e}")
            return GraphSearchResult(
                query=f"multihop: {start_entity}",
                cypher=cypher,
                paths=[],
                entities=[],
                raw_records=[],
            )


    # Cypher keywords / aggregates that are not node variable names
    _CYPHER_NON_VARS = frozenset({
        "count", "sum", "avg", "min", "max", "collect", "distinct",
        "exists", "not", "null", "true", "false", "case", "when",
        "then", "else", "end", "and", "or", "xor", "in", "as",
        "return", "with", "where", "match", "optional", "limit",
        "skip", "order", "by", "asc", "desc", "node", "relationship",
    })

    def _inject_rbac(self, cypher: str, user_role: str) -> str:
        """
        Inject RBAC role filter before RETURN.
        Hardened to reject aggregates, wildcards, and AS-aliased columns
        so the generated WHERE clause never uses a non-node token.
        """
        if user_role == "admin":
            return cypher

        if "RETURN" not in cypher.upper():
            return cypher

        # Already injected guard (prevents double-injection on re-plan cycles)
        if "access_roles" in cypher:
            return cypher

        return_match = re.search(r'(?i)\bRETURN\b\s+(.*)', cypher)
        if not return_match:
            return cypher

        node_vars = []
        for term in return_match.group(1).split(","):
            term = term.strip()

            # Strip LIMIT / ORDER tails that may appear on the last column
            term = re.split(r'\s+(?:LIMIT|ORDER|SKIP)\b', term, flags=re.IGNORECASE)[0].strip()

            # If aliased with AS, take the SOURCE expression not the alias
            as_match = re.split(r'\s+AS\s+', term, flags=re.IGNORECASE)
            source_expr = as_match[0].strip()

            # Take only the variable part (before any dot property access)
            var = source_expr.split("(")[0].split(".")[0].strip()

            # Reject wildcards, functions, aggregates, keywords, and non-identifiers
            if (
                not var
                or var == "*"
                or "(" in source_expr          # function call e.g. count(n)
                or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var)
                or var.lower() in self._CYPHER_NON_VARS
            ):
                continue

            if var not in node_vars:
                node_vars.append(var)

        # Fallback: use generic known variable names
        if not node_vars:
            node_vars = ["e", "n"]

        conditions = " AND ".join(
            f"('{user_role}' IN {v}.access_roles "
            f"OR {v}.access_roles IS NULL "
            f"OR size({v}.access_roles) = 0)"
            for v in node_vars
        )

        rbac_clause = f"WITH * WHERE {conditions}\nRETURN "
        return re.sub(r'(?i)\bRETURN\b', rbac_clause, cypher, count=1)

    def _fallback_search(self, query: str, user_role: str) -> GraphSearchResult:
        """Fallback: parameterized keyword-based entity search when Cypher fails."""
        keywords = query.split()[:5]
        # Prevent empty conditions that would break Cypher syntax
        if not keywords:
            logger.warning(f"Fallback search: empty query after split")
            return GraphSearchResult(query=query, cypher="", paths=[], entities=[], raw_records=[])
        
        conditions = " OR ".join(
            f"toLower(e.name) CONTAINS toLower($kw{i})" for i, _ in enumerate(keywords)
        )
        
        rbac_condition = ""
        if user_role != "admin":
            rbac_condition = f"AND ('{user_role}' IN e.access_roles OR e.access_roles IS NULL OR size(e.access_roles) = 0)"
            
        cypher = f"""
            MATCH (e:Entity)
            WHERE ({conditions})
            {rbac_condition}
            RETURN e.name, e.type, e.context, e.source_doc_ids
            LIMIT {self.cypher_limit}
        """
        params = {f"kw{i}": kw for i, kw in enumerate(keywords)}
        try:
            driver = self._get_driver()
            with driver.session(database=self.database) as session:
                result = session.run(cypher, **params)
                records = [dict(record) for record in result]
            return GraphSearchResult(
                query=query,
                cypher=cypher,
                paths=[],
                entities=self._extract_entities(records),
                raw_records=records,
            )
        except Exception as e:
            logger.error(f"Fallback search also failed: {e}")
            return GraphSearchResult(query=query, cypher=cypher, paths=[], entities=[], raw_records=[])


    def _extract_entities(self, records: list[dict]) -> list[dict]:
        entities = []
        seen = set()
        for record in records:
            for key, value in record.items():
                if isinstance(value, dict) and "name" in value:
                    name = value.get("name", "")
                    if name not in seen:
                        entities.append(value)
                        seen.add(name)
        return entities

    def _extract_paths(self, records: list[dict]) -> list[GraphPath]:
        paths = []
        for record in records:
            if "node_list" in record and "rel_list" in record:
                path_str = " → ".join(
                    f"{n.get('name', '?')}" for n in record["node_list"]
                )
                paths.append(GraphPath(
                    nodes=record["node_list"],
                    relationships=[{"type": r} for r in record["rel_list"]],
                    path_string=path_str,
                ))
        return paths

    def rank_paths_by_centrality(
        self,
        paths: list[GraphPath],
        records: list[dict] = None,
    ) -> list[GraphPath]:
        """
        Rank graph paths by a degree-centrality proxy (FR-27).
        Counts how many times each node name appears across ALL paths, then
        scores each path by the sum of its nodes' appearance counts.
        More central nodes (appear in many paths) rank higher.

        Args:
            paths: List of GraphPath objects to rank.
            records: Optional raw records for additional context.

        Returns:
            Paths sorted by centrality score (descending).
        """
        if not paths:
            return paths

        from collections import Counter

        # Count node appearances across all paths (degree proxy)
        node_counts: Counter = Counter()
        for path in paths:
            for node in path.nodes:
                name = node.get("name", "") if isinstance(node, dict) else str(node)
                if name:
                    node_counts[name] += 1

        # Score each path
        def path_score(path: GraphPath) -> float:
            total = 0.0
            for node in path.nodes:
                name = node.get("name", "") if isinstance(node, dict) else str(node)
                total += node_counts.get(name, 0)
            # Normalize by path length to avoid bias toward longer paths
            length = max(len(path.nodes), 1)
            return total / length

        ranked = sorted(paths, key=path_score, reverse=True)
        logger.info(
            f"Centrality ranking: {len(ranked)} paths ranked by degree centrality"
        )
        return ranked


