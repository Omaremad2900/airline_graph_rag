"""Feature vector embedding-based retrieval using FAISS for vector storage and similarity search."""
from utils.neo4j_connector import Neo4jConnector
from preprocessing.embedding import EmbeddingGenerator
import numpy as np
import faiss
import os
import json
import config


class EmbeddingRetriever:
    """Retrieves information using feature vector embeddings stored in FAISS."""
    
    def __init__(self, connector: Neo4jConnector, embedding_model: EmbeddingGenerator):
        self.connector = connector
        self.embedding_model = embedding_model
        self.dimension = embedding_model.get_dimension()
        
        # Create model-specific names for FAISS index and mapping
        model_safe_name = embedding_model.model_name.replace("/", "_").replace("-", "_")
        self.index_name = f"feature_embedding_{model_safe_name}"
        
        # Directory for storing FAISS indices
        self.faiss_dir = "faiss_indices"
        os.makedirs(self.faiss_dir, exist_ok=True)
        
        # Paths for index and mapping files
        self.index_path = os.path.join(self.faiss_dir, f"{self.index_name}.index")
        self.mapping_path = os.path.join(self.faiss_dir, f"{self.index_name}_mapping.json")
        
        # Initialize FAISS index and ID mapping
        self.index = None
        self.id_mapping = {}  # Maps FAISS ID to Journey feedback_ID
        self._load_index()
    
    def _load_index(self):
        """Load FAISS index and ID mapping from disk if they exist."""
        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            try:
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)
                print(f"✅ Loaded FAISS index from {self.index_path} ({self.index.ntotal} vectors)")
                
                # Load ID mapping
                with open(self.mapping_path, 'r') as f:
                    self.id_mapping = json.load(f)
                print(f"✅ Loaded ID mapping ({len(self.id_mapping)} entries)")
            except Exception as e:
                print(f"⚠️  Error loading FAISS index: {e}")
                self.index = None
                self.id_mapping = {}
        else:
            print(f"ℹ️  No existing FAISS index found. Will create new index when embeddings are generated.")
            self.index = None
            self.id_mapping = {}
    
    def _save_index(self):
        """Save FAISS index and ID mapping to disk."""
        if self.index is not None:
            try:
                # Save FAISS index
                faiss.write_index(self.index, self.index_path)
                print(f"✅ Saved FAISS index to {self.index_path}")
                
                # Save ID mapping
                with open(self.mapping_path, 'w') as f:
                    json.dump(self.id_mapping, f, indent=2)
                print(f"✅ Saved ID mapping to {self.mapping_path}")
            except Exception as e:
                print(f"❌ Error saving FAISS index: {e}")
    
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
    
    def create_feature_embeddings(self, force_recreate: bool = False):
        """
        Create feature vector embeddings for Journey nodes and store them in FAISS.
        For Airline theme without textual features, construct text descriptions
        from numerical properties, then embed them.
        
        Args:
            force_recreate: If True, recreate embeddings even if they already exist
        """
        print(f"Creating feature vector embeddings using model: {self.embedding_model.model_name}")
        print(f"Storing in FAISS index: {self.index_name}")
        
        # Check if embeddings already exist
        if not force_recreate and self.index is not None and self.index.ntotal > 0:
            # Count total journeys in database
            count_query = """
            MATCH (j:Journey)
            WHERE j.feedback_ID IS NOT NULL
            RETURN COUNT(j) as total
            """
            count_result = self.connector.execute_query(count_query)
            db_count = count_result[0].get("total", 0) if count_result else 0
            
            if self.index.ntotal == db_count:
                print(f"✅ Embeddings already exist ({self.index.ntotal} vectors). Use force_recreate=True to regenerate.")
                return self.index.ntotal
        
        # Get journeys with related data (without TAKES relationship)
        query = """
        MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(dep:Airport),
              (f)-[:ARRIVES_AT]->(arr:Airport)
        RETURN j, f, dep, arr
        """
        print("Fetching journeys from Neo4j...")
        results = self.connector.execute_query(query)
        
        if not results:
            print("⚠️  No journeys found in database")
            return 0
        
        print(f"Found {len(results)} journeys. Processing embeddings in batches...")
        
        # Prepare text descriptions first (faster than embedding one-by-one)
        text_descriptions = []
        feedback_ids = []
        
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
            
            feedback_id = j_data.get("feedback_ID")
            if not feedback_id:
                continue
            
            # Create text description from numerical properties
            feature_text = self._create_feature_text(j_data, None, f_data, dep_data, arr_data)
            text_descriptions.append(feature_text)
            feedback_ids.append(feedback_id)
        
        if len(text_descriptions) == 0:
            print("⚠️  No valid journeys found with feedback_ID")
            return 0
        
        # Process embeddings in batches for better performance
        batch_size = 32  # Process 32 at a time
        embeddings_list = []
        total = len(text_descriptions)
        
        print(f"Generating embeddings for {total} journeys (batch size: {batch_size})...")
        for i in range(0, total, batch_size):
            batch_texts = text_descriptions[i:i + batch_size]
            batch_embeddings = self.embedding_model.embed_batch(batch_texts)
            embeddings_list.extend(batch_embeddings)
            
            processed = min(i + batch_size, total)
            print(f"  Processed {processed}/{total} embeddings ({processed*100//total}%)...")
        
        # Create FAISS index
        print(f"Creating FAISS index with {len(embeddings_list)} vectors of dimension {self.dimension}...")
        embeddings_array = np.array(embeddings_list).astype('float32')
        
        # Create FAISS index (using L2 distance, we'll normalize for cosine similarity)
        # For cosine similarity, we normalize vectors and use inner product
        faiss.normalize_L2(embeddings_array)
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for normalized vectors = cosine similarity
        self.index.add(embeddings_array)
        
        # Create ID mapping (FAISS ID -> feedback_ID)
        self.id_mapping = {str(i): feedback_ids[i] for i in range(len(feedback_ids))}
        
        # Save index and mapping
        self._save_index()
        
        print(f"✅ Created {len(embeddings_list)} feature vector embeddings in FAISS index '{self.index_name}'")
        return len(embeddings_list)
    
    def retrieve_by_similarity(self, query: str, top_k: int = 10) -> list:
        """
        Retrieve similar journeys using FAISS similarity search.
        
        Args:
            query: User query text
            top_k: Number of results to return
            
        Returns:
            List of similar journey records with graph data from Neo4j
            
        Raises:
            RuntimeError: If FAISS index is missing or empty
        """
        # Check if index exists
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("FAISS index missing. Run initialize_embeddings.py first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Normalize query vector for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Search in FAISS index
        k = min(top_k, self.index.ntotal)  # Don't request more than available
        distances, indices = self.index.search(query_vector, k)
        
        if len(indices[0]) == 0:
            return []
        
        # Get feedback_IDs from mapping
        feedback_ids = []
        similarity_scores = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # Valid index
                faiss_id = str(int(idx))
                feedback_id = self.id_mapping.get(faiss_id)
                if feedback_id:
                    feedback_ids.append(feedback_id)
                    similarity_scores.append(float(dist))  # Distance is already cosine similarity (inner product of normalized vectors)
        
        if not feedback_ids:
            return []
        
        # Fetch graph data from Neo4j for the similar journeys
        return self._fetch_journey_data(feedback_ids, similarity_scores)
    
    def _fetch_journey_data(self, feedback_ids: list, similarity_scores: list) -> list:
        """
        Fetch journey data from Neo4j for given feedback IDs.
        
        Args:
            feedback_ids: List of Journey feedback_IDs
            similarity_scores: List of similarity scores (in same order)
            
        Returns:
            List of journey records with graph data
        """
        if not feedback_ids:
            return []
        
        # Use parameterized query to avoid type mismatches and injection issues
        query = """
        MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(dep:Airport),
              (f)-[:ARRIVES_AT]->(arr:Airport)
        WHERE j.feedback_ID IN $feedback_ids
        RETURN j, f, dep, arr, j.feedback_ID as feedback_id
        """
        
        results = self.connector.execute_query(query, {"feedback_ids": feedback_ids})
        
        if not results:
            return []
        
        # Create a mapping of feedback_id to similarity score
        score_map = {fid: score for fid, score in zip(feedback_ids, similarity_scores)}
        
        # Process results and add similarity scores
        records = []
        for record in results:
            feedback_id = record.get("feedback_id")
            if feedback_id and feedback_id in score_map:
                # Convert node objects to dictionaries
                flat_record = {}
                
                if "j" in record:
                    j_dict = record["j"]
                    if hasattr(j_dict, 'items'):
                        j_dict = dict(j_dict)
                    flat_record.update(j_dict)
                
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
                
                flat_record["similarity_score"] = score_map[feedback_id]
                records.append(flat_record)
        
        # Sort by similarity score (descending) to maintain order
        records.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        return records
