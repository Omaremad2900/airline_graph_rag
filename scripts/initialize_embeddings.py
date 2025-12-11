"""Script to initialize embeddings in FAISS for embedding-based retrieval."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.neo4j_connector import Neo4jConnector
from preprocessing.embedding import EmbeddingGenerator
from retrieval.embeddings import EmbeddingRetriever
from config import EMBEDDING_MODELS


def initialize_embeddings(model_name: str = None, all_models: bool = False):
    """
    Initialize embeddings for Journey nodes and store them in FAISS.
    
    Args:
        model_name: Embedding model to use (if None and all_models=False, uses first model)
        all_models: If True, initialize embeddings for all configured models
    """
    # Initialize components
    connector = Neo4jConnector()
    if not connector.test_connection():
        print("❌ Failed to connect to Neo4j")
        return
    
    print("✅ Connected to Neo4j")
    
    if all_models:
        # Initialize embeddings for all models
        print(f"Initializing embeddings for {len(EMBEDDING_MODELS)} models...")
        for model_name in EMBEDDING_MODELS.keys():
            print(f"\n{'='*60}")
            print(f"Processing model: {model_name}")
            print(f"{'='*60}")
            
            embedding_model = EmbeddingGenerator(model_name)
            embedding_retriever = EmbeddingRetriever(connector, embedding_model)
            
            count = embedding_retriever.create_feature_embeddings()
            
            if count > 0:
                print(f"✅ Model {model_name} completed: {count} embeddings stored in FAISS")
            else:
                print(f"⚠️  No embeddings created for {model_name}")
    else:
        # Initialize for single model
        if model_name is None:
            model_name = list(EMBEDDING_MODELS.keys())[0]
        
        if model_name not in EMBEDDING_MODELS:
            print(f"❌ Model '{model_name}' not found in EMBEDDING_MODELS")
            print(f"Available models: {list(EMBEDDING_MODELS.keys())}")
            connector.close()
            return
        
        print(f"Initializing embeddings using {model_name}...")
        
        embedding_model = EmbeddingGenerator(model_name)
        embedding_retriever = EmbeddingRetriever(connector, embedding_model)
        
        print("Creating feature vector embeddings for Journey nodes...")
        count = embedding_retriever.create_feature_embeddings()
        
        if count > 0:
            print("✅ Embeddings initialized successfully")
            print(f"   - FAISS index: {embedding_retriever.index_path}")
            print(f"   - ID mapping: {embedding_retriever.mapping_path}")
            print(f"   - Total vectors: {embedding_retriever.index.ntotal}")
        else:
            print("⚠️  No embeddings were created")
    
    connector.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize embeddings in FAISS")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(EMBEDDING_MODELS.keys()),
        help="Embedding model to use (default: first model)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Initialize embeddings for all configured models"
    )
    
    args = parser.parse_args()
    initialize_embeddings(args.model, args.all)
