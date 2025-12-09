"""Script to initialize embeddings in Neo4j for embedding-based retrieval."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.neo4j_connector import Neo4jConnector
from preprocessing.embedding import EmbeddingGenerator
from retrieval.embeddings import EmbeddingRetriever
from config import EMBEDDING_MODELS


def initialize_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initialize embeddings for Journey nodes in Neo4j.
    
    Args:
        model_name: Embedding model to use
    """
    print(f"Initializing embeddings using {model_name}...")
    
    # Initialize components
    connector = Neo4jConnector()
    if not connector.test_connection():
        print("❌ Failed to connect to Neo4j")
        return
    
    print("✅ Connected to Neo4j")
    
    embedding_model = EmbeddingGenerator(model_name)
    embedding_retriever = EmbeddingRetriever(connector, embedding_model)
    
    print("Creating feature embeddings for Journey nodes...")
    embedding_retriever.create_feature_embeddings()
    
    print("✅ Embeddings initialized successfully")
    
    # Create vector index
    dimension = embedding_model.get_dimension()
    print(f"Creating vector index with dimension {dimension}...")
    connector.create_vector_index(
        "journey_feature_embedding_index",
        "Journey",
        "feature_embedding",
        dimension
    )
    
    connector.close()
    print("✅ Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize embeddings in Neo4j")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        choices=list(EMBEDDING_MODELS.keys()),
        help="Embedding model to use"
    )
    
    args = parser.parse_args()
    initialize_embeddings(args.model)

