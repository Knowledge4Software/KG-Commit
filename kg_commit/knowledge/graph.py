import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional
from neo4j import GraphDatabase, Transaction
from .parsers import BaseCommitParser

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GraphKG")


class EntityRule:
    """Defines how a raw dictionary field transforms into a Graph Node Entity."""
    def __init__(self, label: str, primary_key: str, properties_map: Dict[str, str]):
        self.label = label
        self.primary_key = primary_key
        self.properties_map = properties_map


class EdgeRule:
    """Defines how entities connect to one another across the Graph."""
    def __init__(
        self, 
        source_label: str, 
        source_key: str, 
        target_label: str, 
        target_key: str, 
        relationship_type: str,
        is_array_source: bool = False,
        extractor: Optional[Callable[[Any], Dict[str, Any]]] = None
    ):
        self.source_label = source_label
        self.source_key = source_key
        self.target_label = target_label
        self.target_key = target_key
        self.relationship_type = relationship_type
        self.is_array_source = is_array_source
        self.extractor = extractor


class BaseKnowledgeGraph(ABC):
    """Abstract Base Class defining lifecycle pipeline for Knowledge Graph drivers."""

    def __init__(self, uri: str, auth_user: str, auth_pass: str, parser: BaseCommitParser):
        self.driver = GraphDatabase.driver(uri, auth=(auth_user, auth_pass))
        self.parser = parser
        self.entity_rules: List[EntityRule] = []
        self.edge_rules: List[EdgeRule] = []
        self._configure_schema()

    @abstractmethod
    def _configure_schema(self) -> None:
        pass

    def add_entity_rule(self, rule: EntityRule) -> None:
        self.entity_rules.append(rule)

    def add_edge_rule(self, rule: EdgeRule) -> None:
        self.edge_rules.append(rule)

    def close(self) -> None:
        self.driver.close()

    def purge_database(self) -> Dict[str, int]:
        def _purge(tx: Transaction):
            result = tx.run("MATCH (n) DETACH DELETE n")
            return result.consume().counters
        with self.driver.session() as session:
            return session.execute_write(_purge)

    # -----------------------------------------------------------------
    # ADDED: High-Performance Batch Ingestion Method
    # -----------------------------------------------------------------
    def ingest_batch(self, raw_payloads: List[Dict[str, Any]]) -> int:
        """
        Ingests a large collection of raw commits inside a single, high-performance
        reused database session block to completely eliminate connection handshake overhead.
        """
        successful_ingestions = 0
        
        # Open exactly ONE session channel for the entire collection workload
        with self.driver.session() as session:
            for raw_payload in raw_payloads:
                parsed_data = self.parser.parse(raw_payload)
                if not parsed_data:
                    continue
                
                try:
                    # Execute within a managed transaction on the open, reusable session
                    session.execute_write(self._execute_ingestion_transaction, parsed_data)
                    successful_ingestions += 1
                except Exception as e:
                    logger.error(f"Failed to ingest individual commit in batch stream: {e}")
                    
        return successful_ingestions

    def ingest(self, raw_payload: Dict[str, Any]) -> bool:
        """Fallback fallback method for processing single, isolated ad-hoc records."""
        parsed_data = self.parser.parse(raw_payload)
        if not parsed_data:
            return False
        try:
            with self.driver.session() as session:
                session.execute_write(self._execute_ingestion_transaction, parsed_data)
            return True
        except Exception as e:
            logger.error(f"Failed to ingest transaction into Neo4j: {e}")
            return False

    # -----------------------------------------------------------------
    # MODIFIED: Fixed and Batch-Optimized Ingestion Logic
    # -----------------------------------------------------------------
    def _execute_ingestion_transaction(self, tx: Transaction, data: Dict[str, Any]) -> None:
        # Step A: Dynamically MERGE Entity Nodes
        for rule in self.entity_rules:
            if rule.primary_key not in data or data[rule.primary_key] is None:
                continue

            pk_value = data[rule.primary_key]
            props = {target_p: data[src_k] for src_k, target_p in rule.properties_map.items() if src_k in data and data[src_k] is not None}

            query = f"""
            MERGE (e:{rule.label} {{id: $pk_value}})
            ON CREATE SET e += $props
            ON MATCH SET e += $props
            """
            tx.run(query, pk_value=pk_value, props=props)

        # Step B: Dynamically MERGE Vector Edges (Optimized with Cypher UNWIND)
        for rule in self.edge_rules:
            # Fix: Look up the real node ID using the key specified by target_key
            src_id_value = data.get(rule.target_key) 
            if not src_id_value or rule.source_key not in data or data[rule.source_key] is None:
                continue

            raw_items = data[rule.source_key]
            items_list = raw_items if rule.is_array_source else [raw_items]
            
            processed_targets = []
            for item in items_list:
                if not item:
                    continue
                
                if rule.extractor:
                    extracted = rule.extractor(item)
                    if extracted:
                        processed_targets.append({
                            "target_id": extracted["target_id"],
                            "props": extracted.get("properties", {})
                        })
                else:
                    processed_targets.append({
                        "target_id": item,
                        "props": {}
                    })

            if not processed_targets:
                continue

            # Pass the processed targets to a single optimized UNWIND query block
            batch_query = f"""
            MATCH (src:{rule.source_label} {{id: $src_id}})
            UNWIND $targets AS target_item
            MERGE (tgt:{rule.target_label} {{id: target_item.target_id}})
            MERGE (src)-[r:{rule.relationship_type}]->(tgt)
            ON CREATE SET r += target_item.props
            ON MATCH SET r += target_item.props
            """
            tx.run(batch_query, src_id=src_id_value, targets=processed_targets)

class JITCommitKnowledgeGraph(BaseKnowledgeGraph):
    """
    A concrete implementation of the generalizable graph.
    Configures structural graph topology rules optimized for code change analysis.
    """

    # Advanced Tuple Extractor Rule for processing Renamed file structures
    # FIXED: source_key updated to match FilteredCommitParser's payload whitelist
    def rename_extractor(self, tuple_data: tuple) -> Dict[str, Any]:
        if not isinstance(tuple_data, tuple) or len(tuple_data) < 2:
            return {}
        old_path, new_path = tuple_data[0], tuple_data[1]
        return {
            "target_id": new_path,
            "properties": {"renamed_from_historical_path": old_path}
        }
    
    # 1. Define the Tuple/Data Extractor for Copied Files
    def copy_extractor(self, tuple_data: Any) -> Dict[str, Any]:
        """
        Unpacks copy telemetry. 
        Expected format from GitPython/Loader: (old_path, new_path)
        """
        if not isinstance(tuple_data, tuple) or len(tuple_data) < 2:
            return {}
        old_path, new_path = tuple_data[0], tuple_data[1]
        return {
            "target_id": new_path,
            "properties": {
                "copied_from_historical_path": old_path,
                "action": "COPY"
            }
        }


    def _configure_schema(self) -> None:
        # =====================================================================
        # 1. Define Entity Node Mapping Rules
        # =====================================================================
        self.add_entity_rule(EntityRule(
            label="Commit",
            primary_key="commit_id",
            properties_map={"committed_datetime": "committed_at", "message": "message"}
        ))
        
        self.add_entity_rule(EntityRule(
            label="Project",
            primary_key="project",
            properties_map={"project": "name"}
        ))
        
        self.add_entity_rule(EntityRule(
            label="Developer",
            primary_key="author_email",
            properties_map={"author_name": "name", "author_email": "email"}
        ))

        # Explicitly registered File and Branch descriptors so unique IDs map cleanly
        self.add_entity_rule(EntityRule(label="File", 
            primary_key="files_modified_list", 
            properties_map={}))

        self.add_entity_rule(EntityRule(label="Branch", 
            primary_key="containing_branches", 
            properties_map={}))

        # =====================================================================
        # 2. Define Core Edge Relationship Rules
        # =====================================================================
        
        # Connection: Commit -> Project 
        self.add_edge_rule(EdgeRule(
            source_label="Commit", source_key="project",
            target_label="Project", target_key="commit_id", 
            relationship_type="BELONGS_TO"
        ))

        # Connection: Commit -> Developer
        self.add_edge_rule(EdgeRule(
            source_label="Commit", source_key="author_email",
            target_label="Developer", target_key="commit_id", 
            relationship_type="AUTHORED_BY"
        ))

        # Connection: Commit -> Branch
        self.add_edge_rule(EdgeRule(
            source_label="Commit", source_key="containing_branches",
            target_label="Branch", target_key="commit_id",
            relationship_type="PART_OF_BRANCH", is_array_source=True
        ))

        # Connection: Commit -> File (ADDED)
        self.add_edge_rule(EdgeRule(
            source_label="Commit", source_key="files_added_list",
            target_label="File", target_key="commit_id", 
            relationship_type="ADDED", is_array_source=True
        ))

        # Connection: Commit -> File (MODIFIED)
        self.add_edge_rule(EdgeRule(
            source_label="Commit", source_key="files_modified_list",
            target_label="File", target_key="commit_id", 
            relationship_type="MODIFIED", is_array_source=True
        ))

        # Connection: Commit -> File (DELETED)
        self.add_edge_rule(EdgeRule(
            source_label="Commit", source_key="files_deleted_list",
            target_label="File", target_key="commit_id", 
            relationship_type="DELETED", is_array_source=True
        ))


        self.add_edge_rule(EdgeRule(
            source_label="Commit", source_key="files_renamed_list", # <-- FIXED from details to list
            target_label="File", target_key="commit_id", 
            relationship_type="RENAMED_TO",
            is_array_source=True,
            extractor=self.rename_extractor
        ))
        
        # 2. Register the Edge Rule linking Commit -> File
        self.add_edge_rule(EdgeRule(
            source_label="Commit", source_key="files_copied_list",    # Matches the key in FilteredCommitParser
            target_label="File", target_key="commit_id", 
            relationship_type="COPIED_TO",
            is_array_source=True,
            extractor=self.copy_extractor
        ))