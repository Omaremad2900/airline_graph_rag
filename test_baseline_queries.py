"""Comprehensive test script for baseline Cypher query retrieval.

This script tests all baseline query templates across all intent categories.
"""

import json
import sys
from typing import Dict, List, Tuple
from utils.neo4j_connector import Neo4jConnector
from preprocessing.intent_classifier import IntentClassifier
from preprocessing.entity_extractor import EntityExtractor
from retrieval.baseline import BaselineRetriever


def get_test_cases() -> List[Dict]:
    """Get comprehensive test cases for all baseline query templates."""
    return [
        # flight_search intent
        {
            "query": "Find flights from JFK to LAX",
            "intent": "flight_search",
            "expected_templates": ["by_route"],
            "description": "Flight search by route (two airports)"
        },
        {
            "query": "Show me flights departing from JFK",
            "intent": "flight_search",
            "expected_templates": ["by_departure"],
            "description": "Flight search by departure airport"
        },
        {
            "query": "What flights arrive at LAX?",
            "intent": "flight_search",
            "expected_templates": ["by_arrival"],
            "description": "Flight search by arrival airport"
        },
        
        # delay_analysis intent
        {
            "query": "Which flights have delays over 30 minutes?",
            "intent": "delay_analysis",
            "expected_templates": ["flights_with_delays"],
            "description": "Flights with delays above threshold"
        },
        {
            "query": "Show me delays by route",
            "intent": "delay_analysis",
            "expected_templates": ["delays_by_route"],
            "description": "Delays grouped by route"
        },
        {
            "query": "What are the worst delayed flights?",
            "intent": "delay_analysis",
            "expected_templates": ["worst_delayed_flights"],
            "description": "Worst delayed flights (no params)"
        },
        {
            "query": "Find flights with high cancellation risk",
            "intent": "delay_analysis",
            "expected_templates": ["cancelled_flight_patterns"],
            "description": "Cancelled flight patterns"
        },
        {
            "query": "Show me flight reliability statistics",
            "intent": "delay_analysis",
            "expected_templates": ["flight_reliability"],
            "description": "Flight reliability metrics"
        },
        
        # passenger_satisfaction intent
        {
            "query": "Show me journeys with satisfaction below 3",
            "intent": "passenger_satisfaction",
            "expected_templates": ["low_rated_journeys"],
            "description": "Low rated journeys with threshold"
        },
        {
            "query": "What is satisfaction by passenger class?",
            "intent": "passenger_satisfaction",
            "expected_templates": ["satisfaction_by_class"],
            "description": "Satisfaction by class (no params)"
        },
        {
            "query": "Find poor performing flights",
            "intent": "passenger_satisfaction",
            "expected_templates": ["poor_performing_flights"],
            "description": "Poor performing flights"
        },
        {
            "query": "Compare class performance",
            "intent": "passenger_satisfaction",
            "expected_templates": ["class_performance"],
            "description": "Class performance comparison"
        },
        
        # route_analysis intent
        {
            "query": "What are the most popular routes?",
            "intent": "route_analysis",
            "expected_templates": ["popular_routes"],
            "description": "Popular routes (no params)"
        },
        {
            "query": "Show me popular flights",
            "intent": "route_analysis",
            "expected_templates": ["popular_flights"],
            "description": "Popular flights (no params)"
        },
        {
            "query": "What routes have the best performance?",
            "intent": "route_analysis",
            "expected_templates": ["route_performance"],
            "description": "Route performance analysis"
        },
        {
            "query": "Find multi-leg journeys",
            "intent": "route_analysis",
            "expected_templates": ["multi_leg_journeys"],
            "description": "Multi-leg journeys (no params)"
        },
        
        # journey_insights intent
        {
            "query": "Tell me about journey J12345",
            "intent": "journey_insights",
            "expected_templates": ["journey_details"],
            "description": "Journey details by ID"
        },
        {
            "query": "Show me loyalty passenger journeys",
            "intent": "journey_insights",
            "expected_templates": ["loyalty_passenger_journeys"],
            "description": "Loyalty passenger journeys (no params)"
        },
        {
            "query": "Find baggage related journeys",
            "intent": "journey_insights",
            "expected_templates": ["baggage_related_journeys"],
            "description": "Baggage related journeys (no params)"
        },
        
        # performance_metrics intent
        {
            "query": "What are the overall statistics?",
            "intent": "performance_metrics",
            "expected_templates": ["overall_statistics"],
            "description": "Overall statistics (no params)"
        },
        {
            "query": "Show me general flight performance",
            "intent": "performance_metrics",
            "expected_templates": ["general_flight_performance"],
            "description": "General flight performance (no params)"
        },
        {
            "query": "What is the performance of flight 123?",
            "intent": "performance_metrics",
            "expected_templates": ["flight_performance_by_number"],
            "description": "Flight performance by number"
        },
        
        # recommendation intent
        {
            "query": "Recommend the best routes",
            "intent": "recommendation",
            "expected_templates": ["best_routes"],
            "description": "Best routes recommendation (no params)"
        },
        
        # general_question intent
        {
            "query": "What is flight 456?",
            "intent": "general_question",
            "expected_templates": ["flight_info", "flight_info_with_passengers"],
            "description": "Flight info by number"
        },
        
        # flight_performance intent
        {
            "query": "Show me performance for flight 789",
            "intent": "flight_performance",
            "expected_templates": ["flight_performance_by_number", "flight_performance_recent"],
            "description": "Flight performance by number"
        },
        {
            "query": "What are the on-time flights?",
            "intent": "flight_performance",
            "expected_templates": ["on_time_flights"],
            "description": "On-time flights (no params)"
        },
    ]


def test_baseline_queries():
    """Test all baseline query templates."""
    print("=" * 80)
    print("COMPREHENSIVE BASELINE QUERY TESTING")
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
        retriever = BaselineRetriever(connector)
        print("✅ Components initialized")
        print()
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False
    
    # Get test cases
    test_cases = get_test_cases()
    print(f"Testing {len(test_cases)} queries across all intent categories\n")
    print("=" * 80)
    print()
    
    # Statistics
    stats = {
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "no_results": 0,
        "errors": 0,
        "by_intent": {}
    }
    
    # Run tests
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected_intent = test_case["intent"]
        expected_templates = test_case.get("expected_templates", [])
        description = test_case.get("description", "")
        
        print(f"Test {i}/{len(test_cases)}: {query}")
        print(f"Description: {description}")
        print("-" * 80)
        
        try:
            # Classify intent
            intent = classifier.classify(query)
            print(f"Intent: {intent} (expected: {expected_intent})")
            
            if intent != expected_intent:
                print(f"⚠️  Intent mismatch! Expected {expected_intent}, got {intent}")
            
            # Extract entities
            entities = extractor.extract_entities(query)
            entity_summary = {}
            for entity_type, entity_list in entities.items():
                if entity_list:
                    entity_summary[entity_type] = len(entity_list)
            print(f"Entities: {json.dumps(entity_summary) if entity_summary else 'None'}")
            
            # Retrieve results
            results, executed_queries = retriever.retrieve(intent, entities)
            
            print(f"Results: {len(results)} records")
            print(f"Queries executed: {len(executed_queries)}")
            
            # Check executed templates
            executed_template_names = [q["template"] for q in executed_queries]
            print(f"Templates executed: {executed_template_names}")
            
            if expected_templates:
                matched = any(t in executed_template_names for t in expected_templates)
                if matched:
                    print(f"✅ Expected template matched: {[t for t in expected_templates if t in executed_template_names]}")
                else:
                    print(f"⚠️  Expected templates {expected_templates} not found in executed templates")
            
            # Show query details
            if executed_queries:
                for q_info in executed_queries:
                    print(f"  - Template: {q_info['template']}")
                    print(f"    Results: {q_info['result_count']} records")
                    print(f"    Parameters: {json.dumps(q_info['parameters'], indent=6)}")
                    # Show warnings if airport codes are invalid
                    if "warnings" in q_info and q_info["warnings"]:
                        for warning in q_info["warnings"]:
                            print(f"    ⚠️  {warning}")
            
            # Show sample result
            if results:
                print(f"\nSample result (first record):")
                sample = results[0]
                for key, value in list(sample.items())[:8]:  # Show first 8 fields
                    print(f"  {key}: {value}")
                if len(sample) > 8:
                    print(f"  ... and {len(sample) - 8} more fields")
                
                stats["passed"] += 1
            else:
                print("⚠️  No results returned")
                stats["no_results"] += 1
            
            # Update intent stats
            if intent not in stats["by_intent"]:
                stats["by_intent"][intent] = {"total": 0, "passed": 0, "failed": 0}
            stats["by_intent"][intent]["total"] += 1
            if results:
                stats["by_intent"][intent]["passed"] += 1
            else:
                stats["by_intent"][intent]["failed"] += 1
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            stats["errors"] += 1
            stats["failed"] += 1
        
        print()
        print("=" * 80)
        print()
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {stats['total']}")
    print(f"✅ Passed (with results): {stats['passed']}")
    print(f"⚠️  No results: {stats['no_results']}")
    print(f"❌ Failed/Errors: {stats['failed']}")
    print(f"Success rate: {(stats['passed'] / stats['total'] * 100):.1f}%")
    print()
    
    print("Results by Intent:")
    for intent, intent_stats in stats["by_intent"].items():
        success_rate = (intent_stats["passed"] / intent_stats["total"] * 100) if intent_stats["total"] > 0 else 0
        print(f"  {intent}: {intent_stats['passed']}/{intent_stats['total']} ({success_rate:.1f}%)")
    
    print()
    return stats["passed"] > 0


if __name__ == "__main__":
    success = test_baseline_queries()
    sys.exit(0 if success else 1)

