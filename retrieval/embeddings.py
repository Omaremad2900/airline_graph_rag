"""Embedding-based retrieval using semantic similarity."""
from utils.neo4j_connector import Neo4jConnector
from preprocessing.embedding import EmbeddingGenerator
import numpy as np
import config


class EmbeddingRetriever:
    """Retrieves information using semantic similarity search."""
    
    def __init__(self, connector: Neo4jConnector, embedding_model: EmbeddingGenerator):
        self.connector = connector
        self.embedding_model = embedding_model
        self.dimension = embedding_model.get_dimension()
    
    def create_node_embeddings(self, label: str, property_name: str = "embedding"):
        """
        Create embeddings for nodes and store them in Neo4j.
        
        Args:
            label: Node label (e.g., "Flight", "Journey")
            property_name: Property name to store embeddings
        """
        # Get all nodes
        query = f"MATCH (n:{label}) RETURN n LIMIT 1000"
        nodes = self.connector.execute_query(query)
        
        # Generate embeddings and update nodes
        for node in nodes:
            # Create text representation from node properties
            text = self._node_to_text(node["n"], label)
            embedding = self.embedding_model.embed_text(text)
            
            # Update node with embedding
            update_query = f"""
            MATCH (n:{label})
            WHERE id(n) = $node_id
            SET n.{property_name} = $embedding
            """
            # Note: This is simplified - would need proper node ID handling
            pass
    
    def create_feature_embeddings(self):
        """Create feature vector embeddings for Journey nodes."""
        # Get journeys with related data
        query = """
        MATCH (p:Passenger)-[:TAKES]->(j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(dep:Airport),
              (f)-[:ARRIVES_AT]->(arr:Airport)
        RETURN j, p, f, dep, arr
        LIMIT 1000
        """
        results = self.connector.execute_query(query)
        
        embeddings_data = []
        count = 0
        for record in results:
            # Convert node objects to dictionaries if needed
            j_data = record.get("j", {})
            p_data = record.get("p", {})
            f_data = record.get("f", {})
            dep_data = record.get("dep", {})
            arr_data = record.get("arr", {})
            
            # Handle Neo4j Node objects
            if hasattr(j_data, 'items'):
                j_data = dict(j_data)
            if hasattr(p_data, 'items'):
                p_data = dict(p_data)
            if hasattr(f_data, 'items'):
                f_data = dict(f_data)
            if hasattr(dep_data, 'items'):
                dep_data = dict(dep_data)
            if hasattr(arr_data, 'items'):
                arr_data = dict(arr_data)
            
            # Create feature text
            record_dict = {"j": j_data, "p": p_data, "f": f_data, "dep": dep_data, "arr": arr_data}
            feature_text = self._create_feature_text(record_dict)
            embedding = self.embedding_model.embed_text(feature_text)
            
            # Store embedding in Journey node
            feedback_id = j_data.get("feedback_ID")
            if feedback_id:
                update_query = """
                MATCH (j:Journey {feedback_ID: $feedback_id})
                SET j.feature_embedding = $embedding
                """
                self.connector.execute_query(update_query, {
                    "feedback_id": feedback_id,
                    "embedding": embedding
                })
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} embeddings...")
        
        print(f"✅ Created {count} feature embeddings")
    
    def _node_to_text(self, node, label: str) -> str:
        """Convert node properties to text representation."""
        props = dict(node)
        if label == "Flight":
            return f"Flight {props.get('flight_number', '')} with fleet type {props.get('fleet_type_description', '')}"
        elif label == "Journey":
            return f"Journey with {props.get('number_of_legs', 0)} legs, {props.get('actual_flown_miles', 0)} miles, delay {props.get('arrival_delay_minutes', 0)} minutes, food satisfaction {props.get('food_satisfaction_score', 0)}"
        elif label == "Airport":
            return f"Airport {props.get('station_code', '')}"
        elif label == "Passenger":
            return f"Passenger with loyalty level {props.get('loyalty_program_level', '')}"
        return str(props)
    
    def _create_feature_text(self, record: dict) -> str:
        """Create feature text from journey and related data."""
        j = record.get("j", {})
        p = record.get("p", {})
        f = record.get("f", {})
        dep = record.get("dep", {})
        arr = record.get("arr", {})
        
        text_parts = [
            f"Journey from {dep.get('station_code', '')} to {arr.get('station_code', '')}",
            f"Flight {f.get('flight_number', '')}",
            f"Fleet type {f.get('fleet_type_description', '')}",
            f"Passenger class {j.get('passenger_class', '')}",
            f"Loyalty level {p.get('loyalty_program_level', '')}",
            f"Food satisfaction score {j.get('food_satisfaction_score', 0)}",
            f"Arrival delay {j.get('arrival_delay_minutes', 0)} minutes",
            f"Actual miles {j.get('actual_flown_miles', 0)}",
            f"Number of legs {j.get('number_of_legs', 1)}"
        ]
        
        return ". ".join(text_parts)
    
    def retrieve_by_similarity(self, query: str, top_k: int = 10) -> list:
        """
        Retrieve similar journeys using semantic similarity.
        
        Args:
            query: User query text
            top_k: Number of results to return
            
        Returns:
            List of similar journey records
        """
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)
        
        # Check if embeddings exist, if not return empty
        check_query = """
        MATCH (j:Journey)
        WHERE j.feature_embedding IS NOT NULL
        RETURN count(j) as count
        LIMIT 1
        """
        
        try:
            check_result = self.connector.execute_query(check_query)
            if not check_result or check_result[0].get("count", 0) == 0:
                print("No embeddings found. Please run initialize_embeddings.py first.")
                return []
        except Exception:
            pass
        
        # Use manual similarity computation (more reliable)
        return self._manual_similarity_search(query_embedding, top_k)
    
    def _manual_similarity_search(self, query_embedding: list, top_k: int) -> list:
        """Manual similarity search when vector index is not available."""
        # Get all journeys with embeddings
        query = """
        MATCH (p:Passenger)-[:TAKES]->(j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(dep:Airport),
              (f)-[:ARRIVES_AT]->(arr:Airport)
        WHERE j.feature_embedding IS NOT NULL
        RETURN j, p, f, dep, arr, j.feature_embedding as embedding
        LIMIT 500
        """
        
        results = self.connector.execute_query(query)
        
        if not results:
            return []
        
        # Compute cosine similarity
        query_vec = np.array(query_embedding)
        similarities = []
        
        for record in results:
            embedding = record.get("embedding")
            if embedding and isinstance(embedding, list):
                try:
                    vec = np.array(embedding)
                    # Handle dimension mismatch
                    if len(vec) == len(query_vec):
                        norm_query = np.linalg.norm(query_vec)
                        norm_vec = np.linalg.norm(vec)
                        if norm_query > 0 and norm_vec > 0:
                            similarity = np.dot(query_vec, vec) / (norm_query * norm_vec)
                            # Create a flat record for return
                            flat_record = {}
                            if "j" in record:
                                flat_record.update(record["j"])
                            if "p" in record:
                                flat_record.update({f"passenger_{k}": v for k, v in record["p"].items()})
                            if "f" in record:
                                flat_record.update({f"flight_{k}": v for k, v in record["f"].items()})
                            if "dep" in record:
                                flat_record["departure_airport"] = record["dep"].get("station_code", "")
                            if "arr" in record:
                                flat_record["arrival_airport"] = record["arr"].get("station_code", "")
                            flat_record["similarity_score"] = float(similarity)
                            similarities.append((similarity, flat_record))
                except Exception as e:
                    print(f"Error computing similarity: {e}")
                    continue
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in similarities[:top_k]]

