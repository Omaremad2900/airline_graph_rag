"""Baseline retrieval using Cypher queries."""
from utils.neo4j_connector import Neo4jConnector
import config


class BaselineRetriever:
    """Retrieves information from Neo4j using Cypher queries."""
    
    def __init__(self, connector: Neo4jConnector):
        self.connector = connector
        self.query_templates = self._initialize_query_templates()
    
    def _initialize_query_templates(self) -> dict:
        """Initialize Cypher query templates for different intents."""
        return {
            "flight_search": {
                "by_route": """
                    MATCH (f:Flight)-[:DEPARTS_FROM]->(dep:Airport),
                          (f)-[:ARRIVES_AT]->(arr:Airport)
                    WHERE dep.station_code = $departure_code 
                      AND arr.station_code = $arrival_code
                    RETURN f.flight_number as flight_number,
                           f.fleet_type_description as fleet_type,
                           dep.station_code as departure_airport,
                           arr.station_code as arrival_airport
                    LIMIT 20
                """,
                "by_departure": """
                    MATCH (f:Flight)-[:DEPARTS_FROM]->(dep:Airport)
                    WHERE dep.station_code = $departure_code
                    RETURN f.flight_number as flight_number,
                           f.fleet_type_description as fleet_type,
                           dep.station_code as departure_airport
                    LIMIT 20
                """,
                "by_arrival": """
                    MATCH (f:Flight)-[:ARRIVES_AT]->(arr:Airport)
                    WHERE arr.station_code = $arrival_code
                    RETURN f.flight_number as flight_number,
                           f.fleet_type_description as fleet_type,
                           arr.station_code as arrival_airport
                    LIMIT 20
                """
            },
            "delay_analysis": {
                "flights_with_delays": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    WHERE j.arrival_delay_minutes > $min_delay
                    RETURN f.flight_number as flight_number,
                           AVG(j.arrival_delay_minutes) as avg_delay,
                           COUNT(j) as journey_count,
                           MAX(j.arrival_delay_minutes) as max_delay
                    ORDER BY avg_delay DESC
                    LIMIT 20
                """,
                "delays_by_route": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(dep:Airport),
                          (f)-[:ARRIVES_AT]->(arr:Airport)
                    WHERE j.arrival_delay_minutes > 0
                    RETURN dep.station_code as departure,
                           arr.station_code as arrival,
                           AVG(j.arrival_delay_minutes) as avg_delay,
                           COUNT(j) as total_journeys
                    ORDER BY avg_delay DESC
                    LIMIT 20
                """,
                "worst_delayed_flights": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    WHERE j.arrival_delay_minutes IS NOT NULL
                    WITH f, AVG(j.arrival_delay_minutes) as avg_delay, COUNT(j) as count
                    WHERE count >= 5
                    RETURN f.flight_number as flight_number,
                           avg_delay,
                           count
                    ORDER BY avg_delay DESC
                    LIMIT 10
                """
            },
            "passenger_satisfaction": {
                "low_rated_journeys": """
                    MATCH (j:Journey)
                    WHERE j.food_satisfaction_score < $min_score
                    RETURN j.feedback_ID as feedback_id,
                           j.food_satisfaction_score as food_score,
                           j.arrival_delay_minutes as delay_minutes,
                           j.passenger_class as class
                    ORDER BY j.food_satisfaction_score ASC
                    LIMIT 20
                """,
                "satisfaction_by_class": """
                    MATCH (j:Journey)
                    WHERE j.food_satisfaction_score IS NOT NULL
                    RETURN j.passenger_class as passenger_class,
                           AVG(j.food_satisfaction_score) as avg_satisfaction,
                           COUNT(j) as journey_count
                    ORDER BY avg_satisfaction ASC
                """,
                "poor_performing_flights": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    WHERE j.food_satisfaction_score < 3 
                       OR j.arrival_delay_minutes > 30
                    WITH f, 
                         AVG(j.food_satisfaction_score) as avg_satisfaction,
                         AVG(j.arrival_delay_minutes) as avg_delay,
                         COUNT(j) as journey_count
                    WHERE journey_count >= 3
                    RETURN f.flight_number as flight_number,
                           avg_satisfaction,
                           avg_delay,
                           journey_count
                    ORDER BY avg_satisfaction ASC, avg_delay DESC
                    LIMIT 15
                """
            },
            "route_analysis": {
                "popular_routes": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(dep:Airport),
                          (f)-[:ARRIVES_AT]->(arr:Airport)
                    RETURN dep.station_code as departure,
                           arr.station_code as arrival,
                           COUNT(DISTINCT j) as journey_count,
                           AVG(j.actual_flown_miles) as avg_miles
                    ORDER BY journey_count DESC
                    LIMIT 20
                """,
                "route_performance": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(dep:Airport),
                          (f)-[:ARRIVES_AT]->(arr:Airport)
                    WHERE j.arrival_delay_minutes IS NOT NULL
                    RETURN dep.station_code as departure,
                           arr.station_code as arrival,
                           AVG(j.arrival_delay_minutes) as avg_delay,
                           AVG(j.food_satisfaction_score) as avg_satisfaction,
                           COUNT(j) as journey_count
                    ORDER BY avg_delay ASC, avg_satisfaction DESC
                    LIMIT 20
                """,
                "multi_leg_journeys": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    WHERE j.number_of_legs > 1
                    RETURN j.feedback_ID as feedback_id,
                           j.number_of_legs as legs,
                           j.actual_flown_miles as miles,
                           j.arrival_delay_minutes as delay
                    ORDER BY j.number_of_legs DESC
                    LIMIT 20
                """
            },
            "journey_insights": {
                "journey_details": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    WHERE j.feedback_ID = $feedback_id
                    RETURN j.feedback_ID as feedback_id,
                           j.food_satisfaction_score as food_score,
                           j.arrival_delay_minutes as delay,
                           j.actual_flown_miles as miles,
                           j.passenger_class as class,
                           f.flight_number as flight_number
                """,
                "loyalty_passenger_journeys": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    WHERE j.passenger_class IS NOT NULL
                    RETURN j.passenger_class as passenger_class,
                           COUNT(j) as journey_count,
                           AVG(j.food_satisfaction_score) as avg_satisfaction,
                           AVG(j.arrival_delay_minutes) as avg_delay
                    ORDER BY journey_count DESC
                """
            },
            "performance_metrics": {
                "overall_statistics": """
                    MATCH (j:Journey)
                    RETURN AVG(j.food_satisfaction_score) as avg_food_satisfaction,
                           AVG(j.arrival_delay_minutes) as avg_delay,
                           AVG(j.actual_flown_miles) as avg_miles,
                           COUNT(j) as total_journeys
                """,
                "flight_performance": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    WITH f, 
                         AVG(j.food_satisfaction_score) as avg_satisfaction,
                         AVG(j.arrival_delay_minutes) as avg_delay,
                         COUNT(j) as journey_count
                    WHERE journey_count >= 5
                    RETURN f.flight_number as flight_number,
                           f.fleet_type_description as fleet_type,
                           avg_satisfaction,
                           avg_delay,
                           journey_count
                    ORDER BY avg_satisfaction DESC, avg_delay ASC
                    LIMIT 20
                """
            },
            "recommendation": {
                "best_routes": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(dep:Airport),
                          (f)-[:ARRIVES_AT]->(arr:Airport)
                    WHERE j.food_satisfaction_score >= 4 
                      AND j.arrival_delay_minutes <= 15
                    WITH dep, arr, COUNT(j) as good_journeys, COUNT(DISTINCT j) as total
                    WHERE good_journeys >= 10
                    RETURN dep.station_code as departure,
                           arr.station_code as arrival,
                           good_journeys,
                           (toFloat(good_journeys) / total * 100) as satisfaction_rate
                    ORDER BY satisfaction_rate DESC
                    LIMIT 10
                """
            },
            "general_question": {
                "flight_info": """
                    MATCH (f:Flight)-[:DEPARTS_FROM]->(dep:Airport),
                          (f)-[:ARRIVES_AT]->(arr:Airport)
                    WHERE f.flight_number = $flight_number
                    RETURN f.flight_number as flight_number,
                           f.fleet_type_description as fleet_type,
                           dep.station_code as departure,
                           arr.station_code as arrival
                """
            }
        }
    
    def retrieve(self, intent: str, entities: dict) -> tuple:
        """
        Retrieve information based on intent and entities.
        
        Args:
            intent: Classified intent
            entities: Extracted entities
            
        Returns:
            Tuple of (list of retrieved records, list of executed queries with info)
        """
        if intent not in self.query_templates:
            intent = "general_question"
        
        templates = self.query_templates[intent]
        results = []
        executed_queries = []
        
        # For flight_search, prioritize by_route when both airports are present
        if intent == "flight_search" and entities.get("AIRPORT"):
            airports = entities.get("AIRPORT", [])
            airport_codes = [a["value"] for a in airports if a.get("type") == "AIRPORT_CODE"]
            if len(airport_codes) >= 2:
                # Try by_route first
                template_order = ["by_route", "by_departure", "by_arrival"]
            elif len(airport_codes) == 1:
                # Try by_departure first, then by_arrival
                template_order = ["by_departure", "by_arrival", "by_route"]
            else:
                template_order = list(templates.keys())
        else:
            template_order = list(templates.keys())
        
        # Try different query templates based on available entities
        for template_name in template_order:
            if template_name not in templates:
                continue
            query = templates[template_name]
            try:
                parameters = self._build_parameters(entities, template_name)
                if parameters is not None:
                    result = self.connector.execute_query(query, parameters)
                    # Track executed query even if no results
                    executed_queries.append({
                        "template": template_name,
                        "intent": intent,
                        "query": query.strip(),
                        "parameters": parameters,
                        "result_count": len(result) if result else 0
                    })
                    if result:
                        results.extend(result)
            except Exception as e:
                print(f"Error executing query {template_name}: {e}")
                continue
        
        # Remove duplicates based on common keys
        seen = set()
        unique_results = []
        for record in results:
            key = str(sorted(record.items()))
            if key not in seen:
                seen.add(key)
                unique_results.append(record)
        
        return unique_results[:50], executed_queries  # Limit total results
    
    def _build_parameters(self, entities: dict, template_name: str) -> dict:
        """Build query parameters from extracted entities."""
        parameters = {}
        
        # Extract airport codes
        airports = entities.get("AIRPORT", [])
        airport_codes = [a["value"] for a in airports if a.get("type") == "AIRPORT_CODE"]
        airport_names = [a["value"] for a in airports if a.get("type") == "AIRPORT_NAME"]
        
        # Map airport names to codes (simplified - would need a mapping)
        # Prioritize by_route when both airports are present
        if "route" in template_name:
            if len(airport_codes) >= 2:
                parameters["departure_code"] = airport_codes[0]
                parameters["arrival_code"] = airport_codes[1]
            elif len(airport_codes) == 1:
                # If only one airport, can't use by_route
                return None
            else:
                # No airports for route query
                return None
        elif "departure" in template_name:
            if airport_codes:
                # Use first airport as departure
                parameters["departure_code"] = airport_codes[0]
            elif airport_names:
                # Would need airport name to code mapping
                pass
            else:
                # No departure airport found
                return None
        elif "arrival" in template_name:
            if airport_codes:
                # If we have 2 airports, use the second one for arrival
                # Otherwise use the first one
                if len(airport_codes) >= 2:
                    parameters["arrival_code"] = airport_codes[1]
                else:
                    parameters["arrival_code"] = airport_codes[0]
            elif airport_names:
                # Would need airport name to code mapping
                pass
            else:
                # No arrival airport found
                return None
        
        # Extract flight numbers
        flights = entities.get("FLIGHT", [])
        if flights:
            parameters["flight_number"] = flights[0]["value"]
        
        # Extract numbers for thresholds
        numbers = entities.get("NUMBER", [])
        if numbers:
            if "delay" in template_name:
                parameters["min_delay"] = float(numbers[0]["value"])
            elif "satisfaction" in template_name or "score" in template_name:
                parameters["min_score"] = float(numbers[0]["value"])
        else:
            # Provide default thresholds if no number specified
            if "delay" in template_name and "flights_with_delays" in template_name:
                # Default: delays over 15 minutes
                parameters["min_delay"] = 15.0
            elif "satisfaction" in template_name and "low_rated_journeys" in template_name:
                # Default: satisfaction below 3
                parameters["min_score"] = 3.0
        
        # Extract feedback ID
        # This would need more sophisticated entity extraction
        
        # Templates that don't require parameters (can run without entities)
        no_param_templates = [
            "overall_statistics", 
            "popular_routes", 
            "satisfaction_by_class",
            "delays_by_route",
            "worst_delayed_flights",
            "loyalty_passenger_journeys",
            "multi_leg_journeys"
        ]
        
        # If no specific parameters needed, return empty dict
        if not parameters and template_name in no_param_templates:
            return {}
        
        # For queries that require parameters, return None if missing
        # This allows queries without required params to be skipped
        return parameters if parameters else None

