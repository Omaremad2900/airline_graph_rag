"""Comprehensive test script for embeddings-based retrieval.

This script tests semantic similarity search using FAISS indices.
"""

import json
import sys
from typing import Dict, List
from utils.neo4j_connector import Neo4jConnector
from preprocessing.embedding import EmbeddingGenerator
from retrieval.embeddings import EmbeddingRetriever
from config import EMBEDDING_MODELS


def get_test_queries() -> List[Dict]:
    """Get test queries for embeddings retrieval."""
    return [
        {
            "query": "Find flights with long delays",
            "description": "Delay-related query"
        },
        {
            "query": "Show me journeys with poor food satisfaction",
            "description": "Satisfaction-related query"
        },
        {
            "query": "What are flights from New York to Los Angeles?",
            "description": "Route-based query"
        },
        {
            "query": "Find business class passengers with high satisfaction",
            "description": "Class and satisfaction query"
        },
        {
            "query": "Show me on-time flights with good food ratings",
            "description": "Multi-criteria query"
        },
        {
            "query": "What are the most reliable flights?",
            "description": "Reliability query"
        },
        {
            "query": "Find journeys with multiple legs and delays",
            "description": "Complex multi-criteria query"
        },
        {
            "query": "Show me first class passengers on long flights",
            "description": "Class and distance query"
        },
        {
            "query": "What flights have the best passenger experience?",
            "description": "General quality query"
        },
        {
            "query": "Find routes with frequent delays",
            "description": "Route and delay query"
        },
    ]


def test_embeddings_retrieval(model_name: str = None):
    """Test embeddings retrieval for a specific model or all models."""
    print("=" * 80)
    print("COMPREHENSIVE EMBEDDINGS RETRIEVAL TESTING")
    print("=" * 80)
    print()
    
    # Initialize Neo4j connection
    try:
        print("Initializing Neo4j connection...")
        connector = Neo4jConnector()
        if not connector.test_connection():
            print("❌ Failed to connect to Neo4j")
            print("Please ensure Neo4j is running and credentials are correct in .env")
            return False
        print("✅ Connected to Neo4j")
        print()
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    # Determine which models to test
    if model_name:
        if model_name not in EMBEDDING_MODELS:
            print(f"❌ Model '{model_name}' not found in EMBEDDING_MODELS")
            print(f"Available models: {list(EMBEDDING_MODELS.keys())}")
            return False
        models_to_test = [model_name]
    else:
        models_to_test = list(EMBEDDING_MODELS.keys())
    
    print(f"Testing {len(models_to_test)} embedding model(s): {models_to_test}")
    print()
    
    # Initialize retrievers for each model
    retrievers = {}
    for model_name in models_to_test:
        try:
            print(f"Initializing retriever for {EMBEDDING_MODELS[model_name]['name']}...")
            embedding_model = EmbeddingGenerator(model_name)
            retriever = EmbeddingRetriever(connector, embedding_model)
            retrievers[model_name] = retriever
            print(f"✅ Retriever initialized")
        except Exception as e:
            print(f"⚠️  Failed to initialize retriever for {model_name}: {e}")
            print(f"   Make sure FAISS index exists. Run: python scripts/initialize_embeddings.py --model {model_name}")
    
    if not retrievers:
        print("❌ No retrievers initialized. Cannot run tests.")
        return False
    
    print()
    print("=" * 80)
    print()
    
    # Get test queries
    test_queries = get_test_queries()
    print(f"Testing {len(test_queries)} queries\n")
    
    # Statistics
    stats = {
        "total_queries": len(test_queries),
        "by_model": {}
    }
    
    for model_name, retriever in retrievers.items():
        model_display = EMBEDDING_MODELS[model_name]["name"]
        print(f"Testing model: {model_display} ({model_name})")
        print("=" * 80)
        print()
        
        model_stats = {
            "total": len(test_queries),
            "passed": 0,
            "failed": 0,
            "no_results": 0,
            "errors": 0,
            "avg_results": 0,
            "avg_similarity": 0.0
        }
        
        total_results = 0
        total_similarity = 0.0
        
        for i, test_case in enumerate(test_queries, 1):
            query = test_case["query"]
            description = test_case.get("description", "")
            
            print(f"Test {i}/{len(test_queries)}: {query}")
            print(f"Description: {description}")
            print("-" * 80)
            
            try:
                # Retrieve results
                results = retriever.retrieve_by_similarity(query, top_k=10)
                
                if results:
                    print(f"✅ Results: {len(results)} records")
                    
                    # Calculate average similarity
                    similarities = [r.get("similarity_score", 0) for r in results if "similarity_score" in r]
                    if similarities:
                        avg_sim = sum(similarities) / len(similarities)
                        print(f"   Average similarity: {avg_sim:.4f}")
                        total_similarity += avg_sim
                    
                    # Show top 3 results
                    print(f"\nTop 3 results:")
                    for j, result in enumerate(results[:3], 1):
                        print(f"  {j}. Similarity: {result.get('similarity_score', 'N/A'):.4f}")
                        if "flight_number" in result or "flight_flight_number" in result:
                            flight_num = result.get("flight_number") or result.get("flight_flight_number")
                            print(f"     Flight: {flight_num}")
                        if "departure_airport" in result or "departure" in result:
                            dep = result.get("departure_airport") or result.get("departure")
                            arr = result.get("arrival_airport") or result.get("arrival")
                            if dep and arr:
                                print(f"     Route: {dep} → {arr}")
                        if "food_satisfaction_score" in result:
                            print(f"     Food Score: {result['food_satisfaction_score']}")
                        if "arrival_delay_minutes" in result:
                            print(f"     Delay: {result['arrival_delay_minutes']} min")
                        print()
                    
                    total_results += len(results)
                    model_stats["passed"] += 1
                else:
                    print("⚠️  No results returned")
                    model_stats["no_results"] += 1
                
            except RuntimeError as e:
                print(f"❌ Runtime error: {e}")
                print("   Make sure FAISS index exists. Run: python scripts/initialize_embeddings.py")
                model_stats["errors"] += 1
                model_stats["failed"] += 1
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                model_stats["errors"] += 1
                model_stats["failed"] += 1
            
            print()
            print("=" * 80)
            print()
        
        # Calculate averages
        if model_stats["passed"] > 0:
            model_stats["avg_results"] = total_results / model_stats["passed"]
            model_stats["avg_similarity"] = total_similarity / model_stats["passed"]
        
        stats["by_model"][model_name] = model_stats
        
        # Print model summary
        print(f"\nModel Summary: {model_display}")
        print(f"  Total tests: {model_stats['total']}")
        print(f"  ✅ Passed: {model_stats['passed']}")
        print(f"  ⚠️  No results: {model_stats['no_results']}")
        print(f"  ❌ Failed: {model_stats['failed']}")
        print(f"  Success rate: {(model_stats['passed'] / model_stats['total'] * 100):.1f}%")
        if model_stats["avg_results"] > 0:
            print(f"  Average results per query: {model_stats['avg_results']:.1f}")
        if model_stats["avg_similarity"] > 0:
            print(f"  Average similarity score: {model_stats['avg_similarity']:.4f}")
        print()
        print("=" * 80)
        print()
    
    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total queries tested: {stats['total_queries']}")
    print(f"Models tested: {len(stats['by_model'])}")
    print()
    
    print("Results by Model:")
    for model_name, model_stats in stats["by_model"].items():
        model_display = EMBEDDING_MODELS[model_name]["name"]
        success_rate = (model_stats["passed"] / model_stats["total"] * 100) if model_stats["total"] > 0 else 0
        print(f"  {model_display}:")
        print(f"    Success rate: {success_rate:.1f}%")
        print(f"    Avg results: {model_stats['avg_results']:.1f}")
        print(f"    Avg similarity: {model_stats['avg_similarity']:.4f}")
    
    print()
    return any(s["passed"] > 0 for s in stats["by_model"].values())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test embeddings retrieval")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific embedding model to test (default: all models)"
    )
    
    args = parser.parse_args()
    
    success = test_embeddings_retrieval(model_name=args.model)
    sys.exit(0 if success else 1)

