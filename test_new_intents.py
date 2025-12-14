"""Test script for new intent categories and retrieval queries."""
import json
from preprocessing.intent_classifier import IntentClassifier
from preprocessing.entity_extractor import EntityExtractor
from utils.neo4j_connector import Neo4jConnector
from retrieval.baseline import BaselineRetriever


def test_intent_classification():
    """Test intent classification for new intents."""
    print("=" * 80)
    print("TESTING INTENT CLASSIFICATION - NEW INTENTS")
    print("=" * 80)
    
    classifier = IntentClassifier()
    
    test_cases = [
        # Insights-focused intents
        ("What is the performance of flight AA123?", "flight_performance"),
        ("How is flight DL456 performing?", "flight_performance"),
        ("Flight status analysis", "flight_performance"),
        ("Analyze flight reliability", "flight_performance"),
        ("Loyalty program analysis", "loyalty_analysis"),
        ("Frequent flyer program insights", "loyalty_analysis"),
        ("Loyalty metrics by class", "loyalty_analysis"),
        # Existing insights intents (should still work)
        ("Find flights from JFK to LAX", "flight_search"),
        ("Show me delays", "delay_analysis"),
        ("Passenger satisfaction", "passenger_satisfaction"),
        ("Route performance", "route_analysis"),
        ("Journey insights", "journey_insights"),
        ("Performance metrics", "performance_metrics"),
        ("Recommend best routes", "recommendation"),
    ]
    
    print("\nIntent Classification Results:\n")
    all_passed = True
    for query, expected_intent in test_cases:
        predicted = classifier.classify(query)
        status = "[OK]" if predicted == expected_intent else "[FAIL]"
        if predicted != expected_intent:
            all_passed = False
        print(f"{status} Query: '{query}'")
        print(f"   Expected: {expected_intent}")
        print(f"   Predicted: {predicted}")
        print()
    
    if all_passed:
        print("[SUCCESS] All intent classifications passed!")
    else:
        print("[FAILED] Some intent classifications failed!")
    
    return all_passed


def test_entity_extraction():
    """Test entity extraction for new intent queries."""
    print("\n" + "=" * 80)
    print("TESTING ENTITY EXTRACTION - NEW INTENTS")
    print("=" * 80)
    
    extractor = EntityExtractor()
    
    test_cases = [
        ("Find flights from JFK to LAX", ["JFK", "LAX"]),
        ("What is the performance of flight AA123?", ["AA123"]),
        ("Analyze flight DL456 status", ["DL456"]),
        ("Route analysis for BKK to KUL", ["BKK", "KUL"]),
        ("Find flights tomorrow", ["tomorrow"]),
        ("Journey insights for flight on 15/03/24", ["15/03/24"]),
    ]
    
    print("\nEntity Extraction Results:\n")
    all_passed = True
    for query, expected_entities in test_cases:
        entities = extractor.extract_entities(query)
        print(f"Query: '{query}'")
        print(f"Entities: {json.dumps(entities, indent=2)}")
        
        # Check if expected entities are found
        found_entities = []
        if "AIRPORT" in entities:
            found_entities.extend([e["value"] for e in entities["AIRPORT"]])
        if "FLIGHT" in entities:
            found_entities.extend([e["value"] for e in entities["FLIGHT"]])
        if "DATE" in entities:
            found_entities.extend([str(e["value"]) for e in entities["DATE"]])
        
        missing = [e for e in expected_entities if e not in found_entities]
        if missing:
            print(f"   [WARNING] Missing entities: {missing}")
        else:
            print(f"   [OK] All expected entities found")
        print()
    
    return True


def test_retrieval_layer():
    """Test retrieval layer with new intents."""
    print("\n" + "=" * 80)
    print("TESTING RETRIEVAL LAYER - NEW INTENTS")
    print("=" * 80)
    
    try:
        connector = Neo4jConnector()
        if not connector.test_connection():
            print("[ERROR] Cannot connect to Neo4j. Skipping retrieval tests.")
            return False
        
        retriever = BaselineRetriever(connector)
        classifier = IntentClassifier()
        extractor = EntityExtractor()
        
        test_queries = [
            {
                "query": "Find flights from JFK to LAX",
                "intent": "flight_search",
                "entities": {"AIRPORT": [{"value": "JFK", "type": "AIRPORT_CODE"}, {"value": "LAX", "type": "AIRPORT_CODE"}]}
            },
            {
                "query": "What is the performance of flight AA123?",
                "intent": "flight_performance",
                "entities": {"FLIGHT": [{"value": "AA123", "type": "FLIGHT"}]}
            },
            {
                "query": "Show me loyalty program analysis",
                "intent": "loyalty_analysis",
                "entities": {}
            },
            {
                "query": "Analyze flight delays",
                "intent": "delay_analysis",
                "entities": {}
            },
            {
                "query": "Passenger satisfaction by class",
                "intent": "passenger_satisfaction",
                "entities": {}
            },
            {
                "query": "Route performance analysis",
                "intent": "route_analysis",
                "entities": {}
            },
            {
                "query": "Journey insights",
                "intent": "journey_insights",
                "entities": {}
            },
        ]
        
        print("\nRetrieval Test Results:\n")
        all_passed = True
        
        for test_case in test_queries:
            query = test_case["query"]
            expected_intent = test_case["intent"]
            expected_entities = test_case["entities"]
            
            # Classify intent
            predicted_intent = classifier.classify(query)
            
            # Extract entities
            extracted_entities = extractor.extract_entities(query)
            
            print(f"Query: '{query}'")
            print(f"  Intent: {predicted_intent} (expected: {expected_intent})")
            print(f"  Entities: {json.dumps(extracted_entities, indent=4)}")
            
            # Test retrieval
            try:
                # Use extracted intent and entities
                results, executed_queries = retriever.retrieve(predicted_intent, extracted_entities)
                
                print(f"  [OK] Retrieval successful")
                print(f"  Results: {len(results)} records")
                print(f"  Queries executed: {len(executed_queries)}")
                
                if executed_queries:
                    print(f"  Query templates used:")
                    for eq in executed_queries:
                        print(f"    - {eq['template']} ({eq['result_count']} results)")
                
                if results:
                    print(f"  Sample result keys: {list(results[0].keys())}")
                
            except Exception as e:
                print(f"  [ERROR] Retrieval failed: {e}")
                all_passed = False
            
            print()
        
        if all_passed:
            print("[SUCCESS] All retrieval tests passed!")
        else:
            print("[FAILED] Some retrieval tests failed!")
        
        return all_passed
        
    except Exception as e:
        print(f"[ERROR] Error setting up retrieval tests: {e}")
        return False


def test_end_to_end():
    """Test end-to-end pipeline with new intents."""
    print("\n" + "=" * 80)
    print("TESTING END-TO-END PIPELINE - NEW INTENTS")
    print("=" * 80)
    
    classifier = IntentClassifier()
    extractor = EntityExtractor()
    
    try:
        connector = Neo4jConnector()
        if not connector.test_connection():
            print("[ERROR] Cannot connect to Neo4j. Skipping end-to-end tests.")
            return False
        
        retriever = BaselineRetriever(connector)
        
        test_queries = [
            "Find flights from JFK to LAX",
            "What is the performance of flight AA123?",
            "Show me loyalty program analysis",
            "Analyze passenger satisfaction",
            "Route performance metrics",
        ]
        
        print("\nEnd-to-End Test Results:\n")
        
        for query in test_queries:
            print(f"Query: '{query}'")
            print("-" * 80)
            
            # Step 1: Intent Classification
            intent = classifier.classify(query)
            print(f"1. Intent: {intent}")
            
            # Step 2: Entity Extraction
            entities = extractor.extract_entities(query)
            print(f"2. Entities: {json.dumps(entities, indent=2)}")
            
            # Step 3: Retrieval
            try:
                results, executed_queries = retriever.retrieve(intent, entities)
                print(f"3. Retrieval: {len(results)} results from {len(executed_queries)} queries")
                
                if executed_queries:
                    for eq in executed_queries:
                        print(f"   - {eq['template']}: {eq['result_count']} results")
                
                if results and len(results) > 0:
                    print(f"4. Sample result:")
                    sample = results[0]
                    for key, value in list(sample.items())[:5]:  # Show first 5 fields
                        print(f"   {key}: {value}")
                
            except Exception as e:
                print(f"3. Retrieval failed: {e}")
            
            print()
        
        print("[SUCCESS] End-to-end tests completed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error in end-to-end tests: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TESTING NEW INTENT CATEGORIES AND RETRIEVAL QUERIES")
    print("=" * 80)
    
    results = {
        "intent_classification": test_intent_classification(),
        "entity_extraction": test_entity_extraction(),
        "retrieval_layer": test_retrieval_layer(),
        "end_to_end": test_end_to_end()
    }
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[FAILED] Some tests failed!")
    
    return all_passed


if __name__ == "__main__":
    main()
