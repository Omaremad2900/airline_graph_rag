"""Test script for graph retrieval layer.

This script tests both baseline (Cypher queries) and embeddings-based retrieval:
1. Baseline Retrieval - Tests Cypher query templates with different intents and entities
2. Embeddings Retrieval - Tests semantic similarity search
3. Hybrid Retrieval - Tests combining both methods
"""

import json
import sys
from typing import Dict, List


def test_baseline_retrieval():
    """Test baseline retrieval using Cypher queries."""
    print("=" * 60)
    print("TESTING BASELINE RETRIEVAL (Cypher Queries)")
    print("=" * 60)
    
    try:
        from utils.neo4j_connector import Neo4jConnector
        from preprocessing.intent_classifier import IntentClassifier
        from preprocessing.entity_extractor import EntityExtractor
        from retrieval.baseline import BaselineRetriever
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return False
    
    # Initialize components
    try:
        connector = Neo4jConnector()
        if not connector.test_connection():
            print("❌ Failed to connect to Neo4j")
            print("Please ensure Neo4j is running and credentials are correct in .env")
            return False
        print("✅ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    classifier = IntentClassifier()
    extractor = EntityExtractor()
    retriever = BaselineRetriever(connector)
    
    # Test queries from test_queries.json
    try:
        with open('tests/test_queries.json', 'r') as f:
            test_queries = json.load(f)
    except FileNotFoundError:
        print("⚠️  test_queries.json not found, using default test queries")
        test_queries = [
            {"query": "Find flights from JFK to LAX", "intent": "flight_search"},
            {"query": "Which flights have delays over 30 minutes?", "intent": "delay_analysis"},
            {"query": "Show me journeys with low satisfaction", "intent": "passenger_satisfaction"},
            {"query": "What are the most popular routes?", "intent": "route_analysis"},
        ]
    
    print(f"\nTesting {len(test_queries)} queries:\n")
    
    success_count = 0
    for i, test in enumerate(test_queries[:5], 1):  # Test first 5
        query = test.get("query", test)
        if isinstance(query, dict):
            query = query.get("query", "")
        
        print(f"Test {i}: {query}")
        print("-" * 60)
        
        # Classify intent
        intent = classifier.classify(query)
        print(f"Intent: {intent}")
        
        # Extract entities
        entities = extractor.extract_entities(query)
        print(f"Entities: {json.dumps(entities, indent=2) if entities else 'None'}")
        
        # Retrieve results
        try:
            results, executed_queries = retriever.retrieve(intent, entities)
            print(f"Results: {len(results)} records")
            print(f"Queries executed: {len(executed_queries)}")
            
            if executed_queries:
                print("\nExecuted queries:")
                for q_info in executed_queries:
                    print(f"  - {q_info['template']}: {q_info['result_count']} results")
            
            if results:
                print(f"\nSample result (first record):")
                sample = results[0]
                for key, value in list(sample.items())[:5]:  # Show first 5 fields
                    print(f"  {key}: {value}")
                if len(sample) > 5:
                    print(f"  ... and {len(sample) - 5} more fields")
            
            success_count += 1
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n")
    
    connector.close()
    print(f"✅ Baseline retrieval test complete: {success_count}/{min(5, len(test_queries))} successful")
    return success_count > 0


def test_embeddings_retrieval():
    """Test embeddings-based retrieval."""
    print("=" * 60)
    print("TESTING EMBEDDINGS RETRIEVAL")
    print("=" * 60)
    
    try:
        from utils.neo4j_connector import Neo4jConnector
        from preprocessing.embedding import EmbeddingGenerator
        from retrieval.embeddings import EmbeddingRetriever
        from config import EMBEDDING_MODELS
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Initialize components
    try:
        connector = Neo4jConnector()
        if not connector.test_connection():
            print("❌ Failed to connect to Neo4j")
            return False
        print("✅ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    # Test with first available model
    model_name = list(EMBEDDING_MODELS.keys())[0]
    print(f"\nUsing embedding model: {model_name}")
    
    try:
        embedding_model = EmbeddingGenerator(model_name)
        retriever = EmbeddingRetriever(connector, embedding_model)
    except Exception as e:
        print(f"❌ Error initializing embedding model: {e}")
        connector.close()
        return False
    
    # Test queries
    test_queries = [
        "Show me journeys with good food and minimal delays",
        "Find flights with high satisfaction ratings",
        "Which journeys have long delays?",
        "Show me direct flights with good performance"
    ]
    
    print(f"\nTesting {len(test_queries)} semantic queries:\n")
    
    success_count = 0
    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}: {query}")
        print("-" * 60)
        
        try:
            results = retriever.retrieve_by_similarity(query, top_k=5)
            print(f"Results: {len(results)} records")
            
            if results:
                print("\nTop results:")
                for j, result in enumerate(results[:3], 1):  # Show top 3
                    print(f"\n  Result {j} (similarity: {result.get('similarity_score', 'N/A'):.3f}):")
                    # Show key fields
                    key_fields = ['feedback_ID', 'food_satisfaction_score', 'arrival_delay_minutes', 
                                 'departure_airport', 'arrival_airport']
                    for field in key_fields:
                        if field in result:
                            print(f"    {field}: {result[field]}")
            else:
                print("⚠️  No results found. Make sure embeddings are initialized:")
                print("   python scripts/initialize_embeddings.py")
            
            success_count += 1
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n")
    
    connector.close()
    print(f"✅ Embeddings retrieval test complete: {success_count}/{len(test_queries)} successful")
    return success_count > 0


def test_hybrid_retrieval():
    """Test hybrid retrieval (baseline + embeddings)."""
    print("=" * 60)
    print("TESTING HYBRID RETRIEVAL (Baseline + Embeddings)")
    print("=" * 60)
    
    try:
        from utils.neo4j_connector import Neo4jConnector
        from preprocessing.intent_classifier import IntentClassifier
        from preprocessing.entity_extractor import EntityExtractor
        from preprocessing.embedding import EmbeddingGenerator
        from retrieval.baseline import BaselineRetriever
        from retrieval.embeddings import EmbeddingRetriever
        from config import EMBEDDING_MODELS
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Initialize components
    try:
        connector = Neo4jConnector()
        if not connector.test_connection():
            print("❌ Failed to connect to Neo4j")
            return False
        print("✅ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    classifier = IntentClassifier()
    extractor = EntityExtractor()
    baseline_retriever = BaselineRetriever(connector)
    
    # Initialize embedding retriever
    model_name = list(EMBEDDING_MODELS.keys())[0]
    try:
        embedding_model = EmbeddingGenerator(model_name)
        embedding_retriever = EmbeddingRetriever(connector, embedding_model)
    except Exception as e:
        print(f"⚠️  Embedding retriever not available: {e}")
        connector.close()
        return False
    
    # Test query
    query = "Show me flights from JFK with good satisfaction and minimal delays"
    print(f"\nTest Query: {query}")
    print("-" * 60)
    
    # Baseline retrieval
    intent = classifier.classify(query)
    entities = extractor.extract_entities(query)
    print(f"Intent: {intent}")
    print(f"Entities: {json.dumps(entities, indent=2) if entities else 'None'}")
    
    try:
        baseline_results, executed_queries = baseline_retriever.retrieve(intent, entities)
        print(f"\nBaseline Results: {len(baseline_results)} records")
        print(f"Queries executed: {len(executed_queries)}")
    except Exception as e:
        print(f"❌ Baseline retrieval error: {e}")
        baseline_results = []
    
    # Embeddings retrieval
    try:
        embedding_results = embedding_retriever.retrieve_by_similarity(query, top_k=10)
        print(f"Embedding Results: {len(embedding_results)} records")
    except Exception as e:
        print(f"❌ Embedding retrieval error: {e}")
        embedding_results = []
    
    # Combine and deduplicate
    all_results = baseline_results + embedding_results
    seen = set()
    unique_results = []
    for r in all_results:
        key = str(sorted(r.items()))
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    
    print(f"\nCombined Results: {len(unique_results)} unique records")
    print(f"  - From baseline: {len(baseline_results)}")
    print(f"  - From embeddings: {len(embedding_results)}")
    print(f"  - Duplicates removed: {len(all_results) - len(unique_results)}")
    
    connector.close()
    print("\n✅ Hybrid retrieval test complete")
    return True


def test_query_templates():
    """Test that all intent categories have query templates."""
    print("=" * 60)
    print("TESTING QUERY TEMPLATES COVERAGE")
    print("=" * 60)
    
    try:
        from retrieval.baseline import BaselineRetriever
        from utils.neo4j_connector import Neo4jConnector
        from config import INTENTS
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    connector = Neo4jConnector()
    retriever = BaselineRetriever(connector)
    
    print(f"\nIntent categories in config: {len(INTENTS)}")
    print(f"Query templates in baseline: {len(retriever.query_templates)}")
    print("\nCoverage check:")
    print("-" * 60)
    
    all_covered = True
    for intent in INTENTS:
        if intent in retriever.query_templates:
            template_count = len(retriever.query_templates[intent])
            print(f"✅ {intent}: {template_count} template(s)")
        else:
            print(f"❌ {intent}: No templates found")
            all_covered = False
    
    # Count total templates
    total_templates = sum(len(templates) for templates in retriever.query_templates.values())
    print(f"\nTotal query templates: {total_templates}")
    print(f"Requirement: 10+ templates")
    print(f"Status: {'✅ Meets requirement' if total_templates >= 10 else '❌ Below requirement'}")
    
    connector.close()
    return all_covered and total_templates >= 10


def interactive_test():
    """Interactive mode for testing custom queries."""
    print("=" * 60)
    print("INTERACTIVE RETRIEVAL TEST")
    print("=" * 60)
    print("\nEnter queries to test retrieval (type 'quit' to exit)")
    print("Commands:")
    print("  'baseline' - Test baseline only")
    print("  'embeddings' - Test embeddings only")
    print("  'both' - Test hybrid retrieval")
    print("-" * 60)
    
    try:
        from utils.neo4j_connector import Neo4jConnector
        from preprocessing.intent_classifier import IntentClassifier
        from preprocessing.entity_extractor import EntityExtractor
        from preprocessing.embedding import EmbeddingGenerator
        from retrieval.baseline import BaselineRetriever
        from retrieval.embeddings import EmbeddingRetriever
        from config import EMBEDDING_MODELS
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    try:
        connector = Neo4jConnector()
        if not connector.test_connection():
            print("❌ Failed to connect to Neo4j")
            return
        print("✅ Connected to Neo4j\n")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    classifier = IntentClassifier()
    extractor = EntityExtractor()
    baseline_retriever = BaselineRetriever(connector)
    
    # Initialize embedding retriever
    model_name = list(EMBEDDING_MODELS.keys())[0]
    try:
        embedding_model = EmbeddingGenerator(model_name)
        embedding_retriever = EmbeddingRetriever(connector, embedding_model)
        embeddings_available = True
    except Exception as e:
        print(f"⚠️  Embeddings not available: {e}")
        embeddings_available = False
    
    mode = 'both'  # Default mode
    while True:
        query = input("\nEnter query (or command): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        if query.lower() == 'baseline':
            mode = 'baseline'
            print("Mode: Baseline only")
            continue
        elif query.lower() == 'embeddings':
            mode = 'embeddings'
            print("Mode: Embeddings only")
            continue
        elif query.lower() == 'both':
            mode = 'both'
            print("Mode: Hybrid (both)")
            continue
        
        print("\n" + "-" * 60)
        print(f"Query: {query}")
        print("-" * 60)
        
        # Intent and entities
        intent = classifier.classify(query)
        entities = extractor.extract_entities(query)
        print(f"Intent: {intent}")
        print(f"Entities: {json.dumps(entities, indent=2) if entities else 'None'}")
        
        # Baseline
        if mode != 'embeddings':
            try:
                baseline_results, executed_queries = baseline_retriever.retrieve(intent, entities)
                print(f"\nBaseline: {len(baseline_results)} results")
                if executed_queries:
                    for q_info in executed_queries:
                        print(f"  Query: {q_info['template']} → {q_info['result_count']} results")
            except Exception as e:
                print(f"Baseline error: {e}")
        
        # Embeddings
        if mode != 'baseline' and embeddings_available:
            try:
                embedding_results = embedding_retriever.retrieve_by_similarity(query, top_k=5)
                print(f"\nEmbeddings: {len(embedding_results)} results")
                if embedding_results:
                    print("  Top result similarity:", embedding_results[0].get('similarity_score', 'N/A'))
            except Exception as e:
                print(f"Embeddings error: {e}")
        
        print("-" * 60)
    
    connector.close()
    print("\n✅ Interactive test complete")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "baseline":
            test_baseline_retrieval()
        elif mode == "embeddings":
            test_embeddings_retrieval()
        elif mode == "hybrid":
            test_hybrid_retrieval()
        elif mode == "templates":
            test_query_templates()
        elif mode == "interactive":
            interactive_test()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: baseline, embeddings, hybrid, templates, interactive")
    else:
        # Run all tests
        print("Running comprehensive retrieval tests...\n")
        
        print("\n1. Testing Query Templates Coverage")
        print("=" * 60)
        test_query_templates()
        
        print("\n\n2. Testing Baseline Retrieval")
        print("=" * 60)
        test_baseline_retrieval()
        
        print("\n\n3. Testing Embeddings Retrieval")
        print("=" * 60)
        test_embeddings_retrieval()
        
        print("\n\n4. Testing Hybrid Retrieval")
        print("=" * 60)
        test_hybrid_retrieval()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETE")
        print("=" * 60)
        print("\nTo run individual tests:")
        print("  python test_retrieval.py baseline")
        print("  python test_retrieval.py embeddings")
        print("  python test_retrieval.py hybrid")
        print("  python test_retrieval.py templates")
        print("  python test_retrieval.py interactive")

