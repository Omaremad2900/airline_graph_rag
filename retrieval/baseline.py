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
                    MATCH (p:Passenger)-[:TAKES]->(j:Journey)
                    WHERE j.food_satisfaction_score < $min_score
                    RETURN j.feedback_ID as feedback_id,
                           j.food_satisfaction_score as food_score,
                           j.arrival_delay_minutes as delay_minutes,
                           j.passenger_class as class
                    ORDER BY j.food_satisfaction_score ASC
                    LIMIT 20
                """,
                "satisfaction_by_class": """
                    MATCH (p:Passenger)-[:TAKES]->(j:Journey)
                    WHERE j.food_satisfaction_score IS NOT NULL
                    RETURN j.passenger_class as passenger_class,
                           AVG(j.food_satisfaction_score) as avg_satisfaction,
                           COUNT(j) as journey_count
                    ORDER BY avg_satisfaction ASC
                """,
                "poor_performing_flights": """
                    MATCH (j:Journey)-[:ON]->(f:Flight),
                          (p:Passenger)-[:TAKES]->(j)
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
                    MATCH (p:Passenger)-[:TAKES]->(j:Journey)-[:ON]->(f:Flight)
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
                    MATCH (p:Passenger)-[:TAKES]->(j:Journey)-[:ON]->(f:Flight)
                    WHERE j.feedback_ID = $feedback_id
                    RETURN p.record_locator as record_locator,
                           p.loyalty_program_level as loyalty_level,
                           j.feedback_ID as feedback_id,
                           j.food_satisfaction_score as food_score,
                           j.arrival_delay_minutes as delay,
                           j.actual_flown_miles as miles,
                           j.passenger_class as class,
                           f.flight_number as flight_number
                """,
                "loyalty_passenger_journeys": """
                    MATCH (p:Passenger)-[:TAKES]->(j:Journey)-[:ON]->(f:Flight)
                    WHERE p.loyalty_program_level IS NOT NULL
                    RETURN p.loyalty_program_level as loyalty_level,
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
    
    def retrieve(self, intent: str, entities: dict) -> list:
        """
        Retrieve information based on intent and entities.
        
        Args:
            intent: Classified intent
            entities: Extracted entities
            
        Returns:
            List of retrieved records
        """
        if intent not in self.query_templates:
            intent = "general_question"
        
        templates = self.query_templates[intent]
        results = []
        
        # Try different query templates based on available entities
        for template_name, query in templates.items():
            try:
                parameters = self._build_parameters(entities, template_name)
                if parameters is not None:
                    result = self.connector.execute_query(query, parameters)
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
        
        return unique_results[:50]  # Limit total results
    
    def _build_parameters(self, entities: dict, template_name: str) -> dict:
        """Build query parameters from extracted entities."""
        parameters = {}
        
        # Extract airport codes
        airports = entities.get("AIRPORT", [])
        airport_codes = [a["value"] for a in airports if a.get("type") == "AIRPORT_CODE"]
        airport_names = [a["value"] for a in airports if a.get("type") == "AIRPORT_NAME"]
        
        # Map airport names to codes (simplified - would need a mapping)
        if "departure" in template_name or "route" in template_name:
            if airport_codes:
                parameters["departure_code"] = airport_codes[0]
            elif airport_names:
                # Would need airport name to code mapping
                pass
        
        if "arrival" in template_name or "route" in template_name:
            if len(airport_codes) > 1:
                parameters["arrival_code"] = airport_codes[1]
            elif len(airport_codes) == 1 and "departure" not in template_name:
                parameters["arrival_code"] = airport_codes[0]
        
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
        
        # Extract feedback ID
        # This would need more sophisticated entity extraction
        
        # If no specific parameters needed, return empty dict
        if not parameters and template_name in ["overall_statistics", "popular_routes", "satisfaction_by_class"]:
            return {}
        
        return parameters if parameters else None

