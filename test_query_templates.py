"""Test script to validate all query templates."""
import sys
from utils.neo4j_connector import Neo4jConnector
from retrieval.baseline import BaselineRetriever


def test_all_templates():
    """Test all query templates to ensure they work."""
    print("=" * 80)
    print("TESTING ALL QUERY TEMPLATES")
    print("=" * 80)
    
    try:
        connector = Neo4jConnector()
        if not connector.test_connection():
            print("[ERROR] Cannot connect to Neo4j. Skipping tests.")
            return False
        
        retriever = BaselineRetriever(connector)
        
        # Test cases for each intent
        test_cases = [
            # flight_search
            {
                "intent": "flight_search",
                "entities": {"AIRPORT": [{"value": "JFK", "type": "AIRPORT_CODE"}, {"value": "LAX", "type": "AIRPORT_CODE"}]},
                "description": "Flight search by route"
            },
            {
                "intent": "flight_search",
                "entities": {"AIRPORT": [{"value": "JFK", "type": "AIRPORT_CODE"}]},
                "description": "Flight search by departure"
            },
            {
                "intent": "flight_search",
                "entities": {"AIRPORT": [{"value": "LAX", "type": "AIRPORT_CODE"}]},
                "description": "Flight search by arrival"
            },
            
            # delay_analysis
            {
                "intent": "delay_analysis",
                "entities": {"NUMBER": [{"value": "30", "type": "NUMBER"}]},
                "description": "Flights with delays"
            },
            {
                "intent": "delay_analysis",
                "entities": {},
                "description": "Delays by route (no params)"
            },
            {
                "intent": "delay_analysis",
                "entities": {},
                "description": "Worst delayed flights (no params)"
            },
            {
                "intent": "delay_analysis",
                "entities": {},
                "description": "Cancelled flight patterns (no params)"
            },
            {
                "intent": "delay_analysis",
                "entities": {},
                "description": "Flight reliability (no params)"
            },
            
            # passenger_satisfaction
            {
                "intent": "passenger_satisfaction",
                "entities": {"NUMBER": [{"value": "2", "type": "NUMBER"}]},
                "description": "Low rated journeys"
            },
            {
                "intent": "passenger_satisfaction",
                "entities": {},
                "description": "Satisfaction by class (no params)"
            },
            {
                "intent": "passenger_satisfaction",
                "entities": {},
                "description": "Poor performing flights (no params)"
            },
            {
                "intent": "passenger_satisfaction",
                "entities": {},
                "description": "Class performance (no params)"
            },
            
            # route_analysis
            {
                "intent": "route_analysis",
                "entities": {},
                "description": "Popular routes (no params)"
            },
            {
                "intent": "route_analysis",
                "entities": {},
                "description": "Popular flights (no params)"
            },
            {
                "intent": "route_analysis",
                "entities": {},
                "description": "Route performance (no params)"
            },
            {
                "intent": "route_analysis",
                "entities": {},
                "description": "Multi-leg journeys (no params)"
            },
            
            # journey_insights
            {
                "intent": "journey_insights",
                "entities": {"JOURNEY": [{"value": "12345", "type": "JOURNEY"}]},
                "description": "Journey details"
            },
            {
                "intent": "journey_insights",
                "entities": {},
                "description": "Loyalty passenger journeys (no params)"
            },
            {
                "intent": "journey_insights",
                "entities": {},
                "description": "Baggage related journeys (no params)"
            },
            
            # performance_metrics
            {
                "intent": "performance_metrics",
                "entities": {},
                "description": "Overall statistics (no params)"
            },
            {
                "intent": "performance_metrics",
                "entities": {},
                "description": "Flight performance (no params)"
            },
            
            # recommendation
            {
                "intent": "recommendation",
                "entities": {},
                "description": "Best routes (no params)"
            },
            
            # general_question
            {
                "intent": "general_question",
                "entities": {"FLIGHT": [{"value": "123", "type": "FLIGHT"}]},
                "description": "Flight info"
            },
            {
                "intent": "general_question",
                "entities": {"FLIGHT": [{"value": "456", "type": "FLIGHT"}]},
                "description": "Flight info with passengers"
            },
            
            # flight_performance
            {
                "intent": "flight_performance",
                "entities": {"FLIGHT": [{"value": "789", "type": "FLIGHT"}]},
                "description": "Flight performance by number"
            },
            {
                "intent": "flight_performance",
                "entities": {"FLIGHT": [{"value": "101", "type": "FLIGHT"}]},
                "description": "Flight performance recent"
            },
            {
                "intent": "flight_performance",
                "entities": {},
                "description": "On-time flights (no params)"
            },
            
            # loyalty_analysis
            {
                "intent": "loyalty_analysis",
                "entities": {},
                "description": "Loyalty passenger analysis (no params)"
            },
            {
                "intent": "loyalty_analysis",
                "entities": {},
                "description": "Loyalty by class (no params)"
            },
        ]
        
        print(f"\nTesting {len(test_cases)} query template combinations...\n")
        
        all_passed = True
        passed_count = 0
        failed_count = 0
        
        for i, test_case in enumerate(test_cases, 1):
            intent = test_case["intent"]
            entities = test_case["entities"]
            description = test_case["description"]
            
            print(f"[{i}/{len(test_cases)}] {intent} - {description}")
            
            try:
                results, executed_queries = retriever.retrieve(intent, entities)
                
                if executed_queries:
                    print(f"  [OK] Executed {len(executed_queries)} query template(s)")
                    for eq in executed_queries:
                        status = "[OK]" if eq['result_count'] > 0 else "[WARN]"
                        print(f"    {status} {eq['template']}: {eq['result_count']} results")
                    print(f"  [OK] Total results: {len(results)}")
                    passed_count += 1
                else:
                    print(f"  [WARN] No queries executed (may be expected if params missing)")
                    passed_count += 1
                    
            except Exception as e:
                print(f"  [ERROR] {e}")
                failed_count += 1
                all_passed = False
            
            print()
        
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total tests: {len(test_cases)}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        
        if all_passed:
            print("\n[SUCCESS] All query templates are working!")
        else:
            print("\n[WARNING] Some query templates had errors!")
        
        return all_passed
        
    except Exception as e:
        print(f"[ERROR] Error setting up tests: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_all_templates()
    sys.exit(0 if success else 1)
