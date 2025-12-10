"""Test script for the complete Graph-RAG pipeline.

This script tests the full end-to-end pipeline:
1. Preprocessing (Intent Classification + Entity Extraction + Embedding)
2. Retrieval (Baseline Cypher + Embedding-based)
3. LLM Generation
"""

import json
import sys
from typing import Dict, List

# Import components
from preprocessing.intent_classifier import IntentClassifier
from preprocessing.entity_extractor import EntityExtractor
from preprocessing.embedding import EmbeddingGenerator
from utils.neo4j_connector import Neo4jConnector
from retrieval.baseline import BaselineRetriever
from retrieval.embeddings import EmbeddingRetriever
from llm_layer.models import LLMManager
from llm_layer.prompts import build_prompt, get_persona, get_task_instruction


def test_full_pipeline(
    query: str,
    neo4j_uri: str = None,
    neo4j_username: str = None,
    neo4j_password: str = None,
    use_baseline: bool = True,
    use_embeddings: bool = True,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model: str = None,
    verbose: bool = True
) -> Dict:
    """
    Test the complete pipeline from query to response.
    
    Args:
        query: User query to process
        neo4j_uri: Neo4j connection URI (uses config if None)
        neo4j_username: Neo4j username (uses config if None)
        neo4j_password: Neo4j password (uses config if None)
        use_baseline: Whether to use baseline Cypher retrieval
        use_embeddings: Whether to use embedding-based retrieval
        embedding_model: Embedding model name
        llm_model: LLM model name (optional, skips LLM if None)
        verbose: Print detailed output
        
    Returns:
        Dictionary with pipeline results
    """
    results = {
        "query": query,
        "preprocessing": {},
        "retrieval": {},
        "llm": {}
    }
    
    if verbose:
        print("=" * 80)
        print("TESTING FULL GRAPH-RAG PIPELINE")
        print("=" * 80)
        print(f"\nQuery: {query}\n")
    
    # ========== STEP 1: PREPROCESSING ==========
    if verbose:
        print("Step 1: Preprocessing")
        print("-" * 80)
    
    classifier = IntentClassifier()
    extractor = EntityExtractor()
    
    intent = classifier.classify(query)
    entities = extractor.extract_entities(query)
    
    results["preprocessing"] = {
        "intent": intent,
        "entities": entities
    }
    
    if verbose:
        print(f"Intent: {intent}")
        print(f"Entities: {json.dumps(entities, indent=2)}")
    
    # Generate embedding if needed
    embedding = None
    if use_embeddings:
        try:
            generator = EmbeddingGenerator(model_name=embedding_model)
            embedding = generator.embed_text(query)
            results["preprocessing"]["embedding_dimension"] = len(embedding)
            if verbose:
                print(f"Embedding: {len(embedding)} dimensions")
        except Exception as e:
            if verbose:
                print(f"⚠️  Embedding generation failed: {e}")
            use_embeddings = False
    
    # ========== STEP 2: RETRIEVAL ==========
    if verbose:
        print("\nStep 2: Retrieval")
        print("-" * 80)
    
    # Connect to Neo4j
    try:
        connector = Neo4jConnector(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        if not connector.test_connection():
            raise Exception("Failed to connect to Neo4j")
        
        if verbose:
            print("✅ Connected to Neo4j")
    except Exception as e:
        if verbose:
            print(f"❌ Neo4j connection failed: {e}")
        return results
    
    baseline_results = []
    embedding_results = []
    
    # Baseline retrieval
    if use_baseline:
        try:
            baseline_retriever = BaselineRetriever(connector)
            baseline_results = baseline_retriever.retrieve(intent, entities)
            results["retrieval"]["baseline"] = {
                "count": len(baseline_results),
                "results": baseline_results[:5]  # Store first 5 for display
            }
            if verbose:
                print(f"Baseline retrieval: {len(baseline_results)} results")
        except Exception as e:
            if verbose:
                print(f"⚠️  Baseline retrieval failed: {e}")
    
    # Embedding-based retrieval
    if use_embeddings and embedding is not None:
        try:
            embedding_retriever = EmbeddingRetriever(connector, embedding_model)
            embedding_results = embedding_retriever.retrieve_by_similarity(query, top_k=10)
            results["retrieval"]["embeddings"] = {
                "count": len(embedding_results),
                "results": embedding_results[:5]  # Store first 5 for display
            }
            if verbose:
                print(f"Embedding retrieval: {len(embedding_results)} results")
        except Exception as e:
            if verbose:
                print(f"⚠️  Embedding retrieval failed: {e}")
    
    # Combine results
    all_results = baseline_results + embedding_results
    # Deduplicate
    seen = set()
    unique_results = []
    for r in all_results:
        key = str(sorted(r.items()))
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    
    results["retrieval"]["combined"] = {
        "count": len(unique_results),
        "total_unique": len(unique_results)
    }
    
    if verbose:
        print(f"Combined results: {len(unique_results)} unique records")
    
    # ========== STEP 3: LLM GENERATION ==========
    if llm_model and unique_results:
        if verbose:
            print("\nStep 3: LLM Generation")
            print("-" * 80)
        
        try:
            # Format context
            context = json.dumps(unique_results[:30], indent=2)  # Limit context size
            
            # Build prompt
            persona = get_persona()
            task = get_task_instruction()
            prompt = build_prompt(context, persona, task, query)
            
            # Initialize LLM
            llm_manager = LLMManager()
            response = llm_manager.generate_response(llm_model, prompt)
            
            results["llm"] = {
                "model": llm_model,
                "response": response.get("response", ""),
                "metrics": response.get("metrics", {})
            }
            
            if verbose:
                print(f"Model: {llm_model}")
                print(f"Response: {response.get('response', '')[:200]}...")
                if "metrics" in response:
                    print(f"Metrics: {response['metrics']}")
        except Exception as e:
            if verbose:
                print(f"⚠️  LLM generation failed: {e}")
            results["llm"]["error"] = str(e)
    else:
        if verbose:
            if not llm_model:
                print("\nStep 3: LLM Generation (skipped - no model specified)")
            else:
                print("\nStep 3: LLM Generation (skipped - no retrieval results)")
    
    if verbose:
        print("\n" + "=" * 80)
        print("✅ Pipeline Test Complete")
        print("=" * 80)
    
    return results


def test_multiple_queries(queries: List[str], **kwargs):
    """Test multiple queries."""
    print("=" * 80)
    print("TESTING MULTIPLE QUERIES")
    print("=" * 80)
    
    results = []
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(queries)}")
        print(f"{'='*80}")
        result = test_full_pipeline(query, **kwargs)
        results.append(result)
    
    return results


def interactive_test(**kwargs):
    """Interactive testing mode."""
    print("=" * 80)
    print("INTERACTIVE FULL PIPELINE TEST")
    print("=" * 80)
    print("\nEnter queries to test the full pipeline (type 'quit' to exit)\n")
    
    while True:
        query = input("Enter a query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        print()
        test_full_pipeline(query, **kwargs)
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the full Graph-RAG pipeline")
    parser.add_argument("--query", type=str, help="Single query to test")
    parser.add_argument("--file", type=str, help="JSON file with test queries")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline retrieval")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip embedding retrieval")
    parser.add_argument("--embedding-model", type=str, 
                       default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Embedding model name")
    parser.add_argument("--llm-model", type=str, help="LLM model name (optional)")
    parser.add_argument("--neo4j-uri", type=str, help="Neo4j URI (overrides config)")
    parser.add_argument("--neo4j-username", type=str, help="Neo4j username (overrides config)")
    parser.add_argument("--neo4j-password", type=str, help="Neo4j password (overrides config)")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    kwargs = {
        "use_baseline": not args.no_baseline,
        "use_embeddings": not args.no_embeddings,
        "embedding_model": args.embedding_model,
        "llm_model": args.llm_model,
        "neo4j_uri": args.neo4j_uri,
        "neo4j_username": args.neo4j_username,
        "neo4j_password": args.neo4j_password,
        "verbose": not args.quiet
    }
    
    if args.interactive:
        interactive_test(**kwargs)
    elif args.query:
        test_full_pipeline(args.query, **kwargs)
    elif args.file:
        with open(args.file, 'r') as f:
            queries_data = json.load(f)
            queries = [item["query"] for item in queries_data]
        test_multiple_queries(queries, **kwargs)
    else:
        # Default: test with sample queries
        sample_queries = [
            "Which flights have the worst delays?",
            "Show me flights from JFK to LAX",
            "What are the routes with low passenger satisfaction?"
        ]
        print("No query specified. Running with sample queries.")
        print("Use --query 'your query' or --file path/to/queries.json or --interactive")
        print()
        test_multiple_queries(sample_queries, **kwargs)
