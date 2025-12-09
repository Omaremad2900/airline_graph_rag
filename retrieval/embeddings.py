"""Feature vector embedding-based retrieval using text descriptions from numerical properties."""
from utils.neo4j_connector import Neo4jConnector
from preprocessing.embedding import EmbeddingGenerator
import numpy as np
import config


class EmbeddingRetriever:
    """Retrieves information using feature vector embeddings (text descriptions from numerical properties)."""
    
    def __init__(self, connector: Neo4jConnector, embedding_model: EmbeddingGenerator):
        self.connector = connector
        self.embedding_model = embedding_model
        self.dimension = embedding_model.get_dimension()
        # Create model-specific property name for storing embeddings
        model_safe_name = embedding_model.model_name.replace("/", "_").replace("-", "_")
        self.embedding_property = f"feature_embedding_{model_safe_name}"
    
    def _create_feature_text(self, journey_data: dict, passenger_data: dict = None, 
                            flight_data: dict = None, dep_airport: dict = None, 
                            arr_airport: dict = None) -> str:
        """
        Create text description from numerical properties for embedding.
        For Airline theme without textual features, construct text descriptions
        from numerical properties (e.g., "Journey: X, Class: Y, Food: Z, Delay: W").
        
        Args:
            journey_data: Dictionary of Journey node properties
            passenger_data: Dictionary of Passenger node properties (optional)
            flight_data: Dictionary of Flight node properties (optional)
            dep_airport: Dictionary of departure Airport properties (optional)
            arr_airport: Dictionary of arrival Airport properties (optional)
            
        Returns:
            Text description string
        """
        text_parts = []
        
        # Journey identifier
        feedback_id = journey_data.get("feedback_ID", "Unknown")
        text_parts.append(f"Journey {feedback_id}")
        
        # Route information
        if dep_airport and arr_airport:
            dep_code = dep_airport.get("station_code", "")
            arr_code = arr_airport.get("station_code", "")
            if dep_code and arr_code:
                text_parts.append(f"from {dep_code} to {arr_code}")
        
        # Flight information
        if flight_data:
            flight_num = flight_data.get("flight_number", "")
            fleet_type = flight_data.get("fleet_type_description", "")
            if flight_num:
                text_parts.append(f"Flight {flight_num}")
            if fleet_type:
                text_parts.append(f"Fleet type {fleet_type}")
        
        # Passenger class
        passenger_class = journey_data.get("passenger_class", "")
        if passenger_class:
            text_parts.append(f"Class {passenger_class}")
        
        # Loyalty level
        if passenger_data:
            loyalty = passenger_data.get("loyalty_program_level", "")
            if loyalty:
                text_parts.append(f"Loyalty {loyalty}")
        
        # Food satisfaction score
        food_score = journey_data.get("food_satisfaction_score")
        if food_score is not None:
            text_parts.append(f"Food satisfaction {food_score}")
        else:
            text_parts.append("Food satisfaction unknown")
        
        # Arrival delay
        delay = journey_data.get("arrival_delay_minutes")
        if delay is not None:
            if delay < 0:
                text_parts.append(f"Arrived {abs(delay)} minutes early")
            elif delay == 0:
                text_parts.append("Arrived on time")
            else:
                text_parts.append(f"Arrival delay {delay} minutes")
        else:
            text_parts.append("Delay unknown")
        
        # Actual miles flown
        miles = journey_data.get("actual_flown_miles")
        if miles is not None:
            text_parts.append(f"Miles {miles}")
        else:
            text_parts.append("Miles unknown")
        
        # Number of legs
        legs = journey_data.get("number_of_legs")
        if legs is not None:
            if legs == 1:
                text_parts.append("Direct flight")
            else:
                text_parts.append(f"{legs} legs")
        else:
            text_parts.append("Legs unknown")
        
        return ". ".join(text_parts)
    
    def create_feature_embeddings(self):
        """
        Create feature vector embeddings for Journey nodes.
        For Airline theme without textual features, construct text descriptions
        from numerical properties, then embed them.
        """
        print(f"Creating feature vector embeddings using model: {self.embedding_model.model_name}")
        print(f"Storing in property: {self.embedding_property}")
        
        # Get journeys with related data (without TAKES relationship)
        query = """
        MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(dep:Airport),
              (f)-[:ARRIVES_AT]->(arr:Airport)
        RETURN j, f, dep, arr
        """
        results = self.connector.execute_query(query)
        
        count = 0
        for record in results:
            # Convert node objects to dictionaries if needed
            j_data = record.get("j", {})
            f_data = record.get("f", {})
            dep_data = record.get("dep", {})
            arr_data = record.get("arr", {})
            
            # Handle Neo4j Node objects
            if hasattr(j_data, 'items'):
                j_data = dict(j_data)
            if hasattr(f_data, 'items'):
                f_data = dict(f_data)
            if hasattr(dep_data, 'items'):
                dep_data = dict(dep_data)
            if hasattr(arr_data, 'items'):
                arr_data = dict(arr_data)
            
            # Create text description from numerical properties (passenger data is optional)
            feature_text = self._create_feature_text(j_data, None, f_data, dep_data, arr_data)
            
            # Embed the text description using the embedding model
            embedding = self.embedding_model.embed_text(feature_text)
            
            # Store embedding in Journey node with model-specific property name
            feedback_id = j_data.get("feedback_ID")
            if feedback_id:
                # Use parameterized property name with backticks for special characters
                update_query = f"""
                MATCH (j:Journey {{feedback_ID: $feedback_id}})
                SET j.`{self.embedding_property}` = $embedding
                """
                self.connector.execute_query(update_query, {
                    "feedback_id": feedback_id,
                    "embedding": embedding
                })
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} embeddings...")
        
        print(f"✅ Created {count} feature vector embeddings in property '{self.embedding_property}'")
        return count
    
    
    def retrieve_by_similarity(self, query: str, top_k: int = 10) -> list:
        """
        Retrieve similar journeys using feature vector embedding similarity.
        For feature vector embeddings, we embed the query text and compare
        with embedded text descriptions of journeys.
        
        Args:
            query: User query text
            top_k: Number of results to return
            
        Returns:
            List of similar journey records
        """
        # Generate query embedding using the embedding model
        query_embedding = self.embedding_model.embed_text(query)
        
        # Check if embeddings exist, if not return empty
        check_query = f"""
        MATCH (j:Journey)
        WHERE j.`{self.embedding_property}` IS NOT NULL
        RETURN count(j) as count
        LIMIT 1
        """
        
        try:
            check_result = self.connector.execute_query(check_query)
            if not check_result or check_result[0].get("count", 0) == 0:
                print(f"No embeddings found in property '{self.embedding_property}'. Please run initialize_embeddings.py first.")
                return []
        except Exception as e:
            print(f"Error checking embeddings: {e}")
            return []
        
        # Use manual similarity computation with text embeddings
        return self._manual_similarity_search(query_embedding, top_k)
    
    def _manual_similarity_search(self, query_embedding: list, top_k: int) -> list:
        """
        Manual similarity search using text embeddings.
        
        Args:
            query_embedding: Text embedding vector from query
            top_k: Number of results to return
        """
        # Get all journeys with embeddings using model-specific property
        query = f"""
        MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(dep:Airport),
              (f)-[:ARRIVES_AT]->(arr:Airport)
        WHERE j.`{self.embedding_property}` IS NOT NULL
        RETURN j, f, dep, arr, j.`{self.embedding_property}` as embedding
        LIMIT 500
        """
        
        results = self.connector.execute_query(query)
        
        if not results:
            return []
        
        # Compute cosine similarity between query embedding and node embeddings
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
                            # Cosine similarity
                            similarity = np.dot(query_vec, vec) / (norm_query * norm_vec)
                            # Create a flat record for return
                            flat_record = {}
                            if "j" in record:
                                j_dict = record["j"]
                                if hasattr(j_dict, 'items'):
                                    j_dict = dict(j_dict)
                                flat_record.update(j_dict)
                            # Passenger data is optional (TAKES relationship doesn't exist)
                            # if "p" in record:
                            #     p_dict = record["p"]
                            #     if hasattr(p_dict, 'items'):
                            #         p_dict = dict(p_dict)
                            #     flat_record.update({f"passenger_{k}": v for k, v in p_dict.items()})
                            if "f" in record:
                                f_dict = record["f"]
                                if hasattr(f_dict, 'items'):
                                    f_dict = dict(f_dict)
                                flat_record.update({f"flight_{k}": v for k, v in f_dict.items()})
                            if "dep" in record:
                                dep_dict = record["dep"]
                                if hasattr(dep_dict, 'items'):
                                    dep_dict = dict(dep_dict)
                                flat_record["departure_airport"] = dep_dict.get("station_code", "")
                            if "arr" in record:
                                arr_dict = record["arr"]
                                if hasattr(arr_dict, 'items'):
                                    arr_dict = dict(arr_dict)
                                flat_record["arrival_airport"] = arr_dict.get("station_code", "")
                            flat_record["similarity_score"] = float(similarity)
                            similarities.append((similarity, flat_record))
                except Exception as e:
                    print(f"Error computing similarity: {e}")
                    continue
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in similarities[:top_k]]

