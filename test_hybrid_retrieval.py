"""Comprehensive test script for hybrid retrieval (baseline + embeddings).

This script tests combining baseline Cypher queries with embeddings-based
semantic search to evaluate hybrid retrieval performance.
"""

import json
import sys
from typing import Dict, List, Tuple
from utils.neo4j_connector import Neo4jConnector
from preprocessing.intent_classifier import IntentClassifier
from preprocessing.entity_extractor import EntityExtractor
from preprocessing.embedding import EmbeddingGenerator
from retrieval.baseline import BaselineRetriever
from retrieval.embeddings import EmbeddingRetriever
from config import EMBEDDING_MODELS


def get_test_queries() -> List[Dict]:
    """Get test queries for hybrid retrieval."""
    return [
        {
            "query": "Find flights from JFK to LAX with delays",
            "description": "Route + delay query (baseline should find route, embeddings should find delays)"
        },
        {
            "query": "Show me journeys with low satisfaction on flight 123",
            "description": "Flight number + satisfaction (baseline finds flight, embeddings finds satisfaction)"
        },
        {
            "query": "What are the most popular routes with good food ratings?",
            "description": "Route popularity + satisfaction (both methods should contribute)"
        },
        {
            "query": "Find flights with delays over 30 minutes from New York",
            "description": "Delay threshold + location (baseline threshold, embeddings location)"
        },
        {
            "query": "Show me business class passengers with high satisfaction",
            "description": "Class + satisfaction (embeddings should excel)"
        },
        {
            "query": "What is the performance of flight 456?",
            "description": "Flight-specific query (baseline should excel)"
        },
        {
            "query": "Find routes with frequent delays and poor food quality",
            "description": "Multi-criteria query (both methods should contribute)"
        },
        {
            "query": "Show me on-time flights with excellent passenger experience",
            "description": "Performance + quality query (hybrid should combine both)"
        },
        {
            "query": "What are flights from Chicago to Los Angeles with delays?",
            "description": "Specific route + delay (baseline route, embeddings delay context)"
        },
        {
            "query": "Find first class journeys with satisfaction below 3",
            "description": "Class + threshold (baseline threshold, embeddings class)"
        },
    ]


def hybrid_score(record: Dict) -> float:
    """Score for hybrid ranking: embeddings by similarity, baseline gets 0.5"""
    if "similarity_score" in record:
        return float(record["similarity_score"])
    return 0.5


def normalize_record(record: Dict) -> Dict:
    """Normalize record keys for consistent schema"""
    out = dict(record)
    
    # Normalize departure/arrival airport codes
    if "departure_airport" in out and "departure" not in out:
        out["departure"] = out["departure_airport"]
    if "arrival_airport" in out and "arrival" not in out:
        out["arrival"] = out["arrival_airport"]
    
    # Normalize flight number
    if "flight_flight_number" in out and "flight_number" not in out:
        out["flight_number"] = out["flight_flight_number"]
    elif "flight_number" not in out:
        for k, v in out.items():
            if k.startswith("flight_") and "number" in k.lower():
                out["flight_number"] = v
                break
    
    # Normalize feedback_id
    if "feedback_ID" in out and "feedback_id" not in out:
        out["feedback_id"] = out["feedback_ID"]
    
    # Ensure source is set
    if "source" not in out:
        out["source"] = "unknown"
    
    return out


def combine_results(baseline_results: List[Dict], embedding_results: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Combine baseline and embedding results with deduplication and scoring."""
    # Add source metadata
    for r in baseline_results:
        r["source"] = "baseline"
    for r in embedding_results:
        r["source"] = "embeddings"
    
    # Combine
    all_results = baseline_results + embedding_results
    
    # Remove duplicates using stable keys
    seen = set()
    unique_results = []
    for r in all_results:
        if isinstance(r, dict):
            if "flight_number" in r:
                key = f"flight_{r['flight_number']}"
            elif "feedback_id" in r or "feedback_ID" in r:
                fid = r.get("feedback_id") or r.get("feedback_ID")
                key = f"journey_{fid}"
            else:
                key = str(sorted(r.items()))
        else:
            key = str(r)
        
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    
    # Score and sort
    unique_results.sort(key=hybrid_score, reverse=True)
    
    # Normalize
    normalized_results = [normalize_record(r) for r in unique_results]
    
    # Statistics
    stats = {
        "baseline_count": len(baseline_results),
        "embedding_count": len(embedding_results),
        "total_before_dedup": len(all_results),
        "unique_after_dedup": len(unique_results),
        "duplicates_removed": len(all_results) - len(unique_results),
        "baseline_in_final": sum(1 for r in normalized_results if r.get("source") == "baseline"),
        "embedding_in_final": sum(1 for r in normalized_results if r.get("source") == "embeddings"),
    }
    
    return normalized_results, stats


def test_hybrid_retrieval(embedding_model_name: str = None):
    """Test hybrid retrieval combining baseline and embeddings."""
    print("=" * 80)
    print("COMPREHENSIVE HYBRID RETRIEVAL TESTING")
    print("=" * 80)
    print()
    
    # Initialize components
    try:
        print("Initializing components...")
        connector = Neo4jConnector()
        if not connector.test_connection():
            print("❌ Failed to connect to Neo4j")
            print("Please ensure Neo4j is running and credentials are correct in .env")
            return False
        print("✅ Connected to Neo4j")
        
        classifier = IntentClassifier()
        extractor = EntityExtractor()
        baseline_retriever = BaselineRetriever(connector)
        print("✅ Baseline retriever initialized")
        
        # Initialize embedding retriever
        if embedding_model_name is None:
            embedding_model_name = list(EMBEDDING_MODELS.keys())[0]
        
        if embedding_model_name not in EMBEDDING_MODELS:
            print(f"❌ Model '{embedding_model_name}' not found")
            print(f"Available models: {list(EMBEDDING_MODELS.keys())}")
            return False
        
        print(f"Initializing embedding retriever for {EMBEDDING_MODELS[embedding_model_name]['name']}...")
        embedding_model = EmbeddingGenerator(embedding_model_name)
        embedding_retriever = EmbeddingRetriever(connector, embedding_model)
        print("✅ Embedding retriever initialized")
        print()
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Get test queries
    test_queries = get_test_queries()
    print(f"Testing {len(test_queries)} queries with hybrid retrieval\n")
    print("=" * 80)
    print()
    
    # Statistics
    stats = {
        "total": len(test_queries),
        "passed": 0,
        "failed": 0,
        "baseline_only": 0,
        "embedding_only": 0,
        "both_contributed": 0,
        "avg_baseline_results": 0,
        "avg_embedding_results": 0,
        "avg_hybrid_results": 0,
        "avg_duplicates": 0,
    }
    
    total_baseline = 0
    total_embedding = 0
    total_hybrid = 0
    total_duplicates = 0
    
    # Run tests
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        description = test_case.get("description", "")
        
        print(f"Test {i}/{len(test_queries)}: {query}")
        print(f"Description: {description}")
        print("-" * 80)
        
        try:
            # Preprocessing
            intent = classifier.classify(query)
            entities = extractor.extract_entities(query)
            print(f"Intent: {intent}")
            entity_summary = {k: len(v) for k, v in entities.items() if v}
            print(f"Entities: {json.dumps(entity_summary) if entity_summary else 'None'}")
            
            # Baseline retrieval
            print("\n[Baseline Retrieval]")
            baseline_results, executed_queries = baseline_retriever.retrieve(intent, entities)
            print(f"  Results: {len(baseline_results)} records")
            print(f"  Queries executed: {len(executed_queries)}")
            if executed_queries:
                for q_info in executed_queries:
                    print(f"    - {q_info['template']}: {q_info['result_count']} results")
            
            # Embedding retrieval
            print("\n[Embedding Retrieval]")
            try:
                embedding_results = embedding_retriever.retrieve_by_similarity(query, top_k=10)
                print(f"  Results: {len(embedding_results)} records")
                if embedding_results:
                    avg_sim = sum(r.get("similarity_score", 0) for r in embedding_results) / len(embedding_results)
                    print(f"  Average similarity: {avg_sim:.4f}")
            except RuntimeError as e:
                print(f"  ⚠️  {e}")
                embedding_results = []
            except Exception as e:
                print(f"  ❌ Error: {e}")
                embedding_results = []
            
            # Combine results
            print("\n[Hybrid Combination]")
            hybrid_results, combo_stats = combine_results(baseline_results, embedding_results)
            print(f"  Baseline results: {combo_stats['baseline_count']}")
            print(f"  Embedding results: {combo_stats['embedding_count']}")
            print(f"  Total before dedup: {combo_stats['total_before_dedup']}")
            print(f"  Unique after dedup: {combo_stats['unique_after_dedup']}")
            print(f"  Duplicates removed: {combo_stats['duplicates_removed']}")
            print(f"  Baseline in final: {combo_stats['baseline_in_final']}")
            print(f"  Embedding in final: {combo_stats['embedding_in_final']}")
            
            # Analyze contribution
            if combo_stats['baseline_in_final'] > 0 and combo_stats['embedding_in_final'] > 0:
                print("  ✅ Both methods contributed to final results")
                stats["both_contributed"] += 1
            elif combo_stats['baseline_in_final'] > 0:
                print("  ⚠️  Only baseline contributed")
                stats["baseline_only"] += 1
            elif combo_stats['embedding_in_final'] > 0:
                print("  ⚠️  Only embeddings contributed")
                stats["embedding_only"] += 1
            
            # Show top results
            if hybrid_results:
                print(f"\nTop 5 hybrid results (sorted by score):")
                for j, result in enumerate(hybrid_results[:5], 1):
                    source = result.get("source", "unknown")
                    score = result.get("similarity_score", 0.5)
                    print(f"  {j}. [{source}] Score: {score:.4f}")
                    if "flight_number" in result:
                        print(f"     Flight: {result['flight_number']}")
                    if "departure" in result and "arrival" in result:
                        print(f"     Route: {result['departure']} → {result['arrival']}")
                    if "food_satisfaction_score" in result:
                        print(f"     Food Score: {result['food_satisfaction_score']}")
                    print()
                
                total_baseline += len(baseline_results)
                total_embedding += len(embedding_results)
                total_hybrid += len(hybrid_results)
                total_duplicates += combo_stats['duplicates_removed']
                stats["passed"] += 1
            else:
                print("  ⚠️  No hybrid results")
                stats["failed"] += 1
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            stats["failed"] += 1
        
        print()
        print("=" * 80)
        print()
    
    # Calculate averages
    if stats["passed"] > 0:
        stats["avg_baseline_results"] = total_baseline / stats["passed"]
        stats["avg_embedding_results"] = total_embedding / stats["passed"]
        stats["avg_hybrid_results"] = total_hybrid / stats["passed"]
        stats["avg_duplicates"] = total_duplicates / stats["passed"]
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {stats['total']}")
    print(f"✅ Passed: {stats['passed']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"Success rate: {(stats['passed'] / stats['total'] * 100):.1f}%")
    print()
    print("Contribution Analysis:")
    print(f"  Both methods contributed: {stats['both_contributed']}")
    print(f"  Baseline only: {stats['baseline_only']}")
    print(f"  Embeddings only: {stats['embedding_only']}")
    print()
    print("Average Results:")
    print(f"  Baseline: {stats['avg_baseline_results']:.1f}")
    print(f"  Embeddings: {stats['avg_embedding_results']:.1f}")
    print(f"  Hybrid (after dedup): {stats['avg_hybrid_results']:.1f}")
    print(f"  Duplicates removed: {stats['avg_duplicates']:.1f}")
    print()
    
    return stats["passed"] > 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test hybrid retrieval")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model to use (default: first available)"
    )
    
    args = parser.parse_args()
    
    success = test_hybrid_retrieval(embedding_model_name=args.model)
    sys.exit(0 if success else 1)

