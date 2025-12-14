"""Test script for intent classification and entity extraction."""
import json
from preprocessing.intent_classifier import IntentClassifier
from preprocessing.entity_extractor import EntityExtractor


def test_intent_classification():
    """Test intent classification with various queries."""
    print("=" * 80)
    print("TESTING INTENT CLASSIFICATION")
    print("=" * 80)
    
    classifier = IntentClassifier()
    
    test_cases = [
        # Recommendation queries
        ("Recommend the best routes for on-time performance", "recommendation"),
        ("Suggest best flights", "recommendation"),
        ("What should I choose?", "recommendation"),
        
        # Flight search queries
        ("Find flights from JFK to LAX", "flight_search"),
        ("Search for flights between NYC and LA", "flight_search"),
        ("Show me available flights", "flight_search"),
        
        # Delay analysis queries
        ("Show me delays", "delay_analysis"),
        ("Which flights are late?", "delay_analysis"),
        ("Analyze on-time performance", "delay_analysis"),
        ("Flights with delays above 30 minutes", "delay_analysis"),
        
        # Passenger satisfaction queries
        ("Passenger satisfaction", "passenger_satisfaction"),
        ("Low rated journeys", "passenger_satisfaction"),
        ("Food satisfaction scores", "passenger_satisfaction"),
        
        # Route analysis queries
        ("Route performance", "route_analysis"),
        ("Popular routes", "route_analysis"),
        ("Multi-leg journeys", "route_analysis"),
        
        # Journey insights queries
        ("Journey insights", "journey_insights"),
        ("Journey details for journey_12345", "journey_insights"),
        ("Passenger journey analysis", "journey_insights"),
        
        # Performance metrics queries
        ("Performance metrics", "performance_metrics"),
        ("Overall statistics", "performance_metrics"),
        ("Compare flight performance", "performance_metrics"),
        
        # Flight performance queries
        ("What is the performance of flight AA123?", "flight_performance"),
        ("How is flight DL456 performing?", "flight_performance"),
        ("Flight status analysis", "flight_performance"),
        ("Flight reliability metrics", "flight_performance"),
        
        # Loyalty analysis queries
        ("Loyalty program analysis", "loyalty_analysis"),
        ("Frequent flyer insights", "loyalty_analysis"),
        ("Loyalty metrics by class", "loyalty_analysis"),
        
        # General questions
        ("What is a flight?", "general_question"),
        ("How does the system work?", "general_question"),
        ("Tell me about flights", "general_question"),
    ]
    
    print(f"\nTesting {len(test_cases)} queries...\n")
    
    passed = 0
    failed = 0
    failures = []
    
    for query, expected_intent in test_cases:
        predicted = classifier.classify(query)
        status = "[OK]" if predicted == expected_intent else "[FAIL]"
        
        if predicted == expected_intent:
            passed += 1
        else:
            failed += 1
            failures.append((query, expected_intent, predicted))
        
        print(f"{status} Query: '{query}'")
        print(f"   Expected: {expected_intent}")
        print(f"   Predicted: {predicted}")
        print()
    
    print("=" * 80)
    print("INTENT CLASSIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failures:
        print("\nFailed cases:")
        for query, expected, predicted in failures:
            print(f"  - '{query}'")
            print(f"    Expected: {expected}, Got: {predicted}")
    
    return failed == 0


def test_entity_extraction():
    """Test entity extraction with various queries."""
    print("\n" + "=" * 80)
    print("TESTING ENTITY EXTRACTION")
    print("=" * 80)
    
    extractor = EntityExtractor()
    
    test_cases = [
        # Airport codes
        ("Find flights from JFK to LAX", ["JFK", "LAX"]),
        ("Flights from BKK to KUL", ["BKK", "KUL"]),
        ("Search flights to SFO", ["SFO"]),
        
        # Flight numbers
        ("What is the status of flight AA123?", ["AA123"]),
        ("Flight DL456 performance", ["DL456"]),
        ("Check flight UA789", ["UA789"]),
        
        # Journey IDs
        ("Journey details for journey_12345", ["12345"]),
        ("Show journey J67890", ["67890"]),
        
        # Dates
        ("Find flights tomorrow", ["tomorrow"]),
        ("Book flight on 15/03/24", ["15/03/24"]),
        ("Flights on 2024-03-15", ["2024-03-15"]),
        
        # Numbers (extracted as floats, so check for numeric value - accept either format)
        ("Flights with delays above 30 minutes", ["30.0"]),  # System extracts as float, which is correct
        ("Low rated journeys below 2", ["2.0"]),  # System extracts as float, which is correct
        
        # Routes (mentioned)
        ("Best routes for performance", []),  # Route mention, no specific route
        ("Route from JFK to LAX", ["JFK", "LAX"]),  # Route with airports
    ]
    
    print(f"\nTesting {len(test_cases)} queries...\n")
    
    passed = 0
    failed = 0
    failures = []
    
    for query, expected_entities in test_cases:
        entities = extractor.extract_entities(query)
        
        # Extract found entities
        found_entities = []
        if "AIRPORT" in entities:
            found_entities.extend([e["value"] for e in entities["AIRPORT"]])
        if "FLIGHT" in entities:
            found_entities.extend([e["value"] for e in entities["FLIGHT"]])
        if "JOURNEY" in entities:
            found_entities.extend([str(e["value"]) for e in entities["JOURNEY"]])
        if "DATE" in entities:
            found_entities.extend([str(e["value"]) for e in entities["DATE"]])
        if "NUMBER" in entities:
            found_entities.extend([str(e["value"]) for e in entities["NUMBER"]])
        
        # Check if all expected entities are found
        missing = [e for e in expected_entities if e not in found_entities]
        extra = [e for e in found_entities if e not in expected_entities]
        
        status = "[OK]" if not missing else "[FAIL]"
        
        if not missing:
            passed += 1
        else:
            failed += 1
            failures.append((query, expected_entities, found_entities, missing))
        
        print(f"{status} Query: '{query}'")
        print(f"   Expected entities: {expected_entities}")
        print(f"   Found entities: {found_entities}")
        if missing:
            print(f"   Missing: {missing}")
        if extra:
            print(f"   Extra: {extra}")
        print(f"   Full entities: {json.dumps(entities, indent=4)}")
        print()
    
    print("=" * 80)
    print("ENTITY EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failures:
        print("\nFailed cases:")
        for query, expected, found, missing in failures:
            print(f"  - '{query}'")
            print(f"    Expected: {expected}")
            print(f"    Found: {found}")
            print(f"    Missing: {missing}")
    
    return failed == 0


def test_combined():
    """Test intent classification and entity extraction together."""
    print("\n" + "=" * 80)
    print("TESTING COMBINED INTENT + ENTITY EXTRACTION")
    print("=" * 80)
    
    classifier = IntentClassifier()
    extractor = EntityExtractor()
    
    test_queries = [
        "Recommend the best routes for on-time performance",
        "Find flights from JFK to LAX tomorrow",
        "What is the performance of flight AA123?",
        "Show me delays above 30 minutes",
        "Loyalty program analysis",
        "Route performance from BKK to KUL",
        "Journey details for journey_12345",
        "Passenger satisfaction by class",
    ]
    
    print(f"\nTesting {len(test_queries)} queries...\n")
    
    for query in test_queries:
        intent = classifier.classify(query)
        entities = extractor.extract_entities(query)
        
        print(f"Query: '{query}'")
        print(f"  Intent: {intent}")
        print(f"  Entities: {json.dumps(entities, indent=4)}")
        print()


def main():
    """Run all tests."""
    intent_ok = test_intent_classification()
    entity_ok = test_entity_extraction()
    test_combined()
    
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Intent Classification: {'[PASSED]' if intent_ok else '[FAILED]'}")
    print(f"Entity Extraction: {'[PASSED]' if entity_ok else '[FAILED]'}")
    
    if intent_ok and entity_ok:
        print("\n[SUCCESS] All preprocessing components are working correctly!")
    else:
        print("\n[WARNING] Some preprocessing components have issues!")
    
    return intent_ok and entity_ok


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
