"""Baseline retrieval using Cypher queries."""
from utils.neo4j_connector import Neo4jConnector
import logging

logger = logging.getLogger(__name__)
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
                """,
                "cancelled_flight_patterns": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    WHERE j.arrival_delay_minutes > 60
                    WITH f,
                         COUNT(j) as high_delay_count,
                         AVG(j.arrival_delay_minutes) as avg_delay
                    WHERE high_delay_count >= 3
                    RETURN f.flight_number as flight_number,
                           high_delay_count,
                           avg_delay,
                           'High delay rate - potential cancellation risk' as note
                    ORDER BY avg_delay DESC
                    LIMIT 15
                """,
                "flight_reliability": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    WITH f,
                         COUNT(j) as total_journeys,
                         SUM(CASE WHEN j.arrival_delay_minutes > 60 THEN 1 ELSE 0 END) as high_delay_count
                    WHERE total_journeys >= 5
                    RETURN f.flight_number as flight_number,
                           total_journeys,
                           high_delay_count,
                           (toFloat(high_delay_count) / total_journeys * 100) as high_delay_percentage
                    ORDER BY high_delay_percentage DESC
                    LIMIT 20
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
                """,
                "class_performance": """
                    MATCH (j:Journey)
                    WHERE j.passenger_class IS NOT NULL
                    RETURN j.passenger_class as passenger_class,
                           COUNT(j) as journey_count,
                           AVG(j.food_satisfaction_score) as avg_satisfaction,
                           AVG(j.arrival_delay_minutes) as avg_delay
                    ORDER BY avg_satisfaction DESC
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
                "popular_flights": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(dep:Airport),
                          (f)-[:ARRIVES_AT]->(arr:Airport)
                    RETURN f.flight_number as flight_number,
                           dep.station_code as departure,
                           arr.station_code as arrival,
                           COUNT(j) as journey_count,
                           AVG(j.food_satisfaction_score) as avg_satisfaction
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
                """,
                "baggage_related_journeys": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    WHERE j.food_satisfaction_score IS NOT NULL
                    RETURN j.feedback_ID as feedback_id,
                           j.passenger_class as passenger_class,
                           j.food_satisfaction_score as satisfaction,
                           f.flight_number as flight_number,
                           j.actual_flown_miles as miles
                    ORDER BY j.feedback_ID DESC
                    LIMIT 20
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
                    WITH dep, arr, 
                         COUNT(j) as total_journeys,
                         SUM(CASE WHEN j.food_satisfaction_score >= 4 AND j.arrival_delay_minutes <= 15 THEN 1 ELSE 0 END) as good_journeys
                    WHERE total_journeys >= 10 AND good_journeys >= 10
                    RETURN dep.station_code as departure,
                           arr.station_code as arrival,
                           good_journeys,
                           total_journeys,
                           (toFloat(good_journeys) / total_journeys * 100) as satisfaction_rate
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
                """,
                "flight_info_with_passengers": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(dep:Airport)
                    WHERE f.flight_number = $flight_number
                    RETURN DISTINCT f.flight_number as flight_number,
                           dep.station_code as departure_airport,
                           COUNT(DISTINCT j.feedback_ID) as passenger_count,
                           AVG(j.food_satisfaction_score) as avg_satisfaction
                    LIMIT 1
                """
            },
            "flight_performance": {
                "flight_performance_by_number": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(dep:Airport),
                          (f)-[:ARRIVES_AT]->(arr:Airport)
                    WHERE f.flight_number = $flight_number
                    WITH f, dep, arr,
                         AVG(j.arrival_delay_minutes) as avg_delay,
                         COUNT(j) as journey_count,
                         SUM(CASE WHEN j.arrival_delay_minutes <= 15 THEN 1 ELSE 0 END) as on_time_count
                    RETURN f.flight_number as flight_number,
                           dep.station_code as departure,
                           arr.station_code as arrival,
                           avg_delay,
                           journey_count,
                           on_time_count
                    LIMIT 1
                """,
                "flight_performance_recent": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(dep:Airport),
                          (f)-[:ARRIVES_AT]->(arr:Airport)
                    WHERE f.flight_number = $flight_number
                    RETURN f.flight_number as flight_number,
                           dep.station_code as departure,
                           arr.station_code as arrival,
                           j.arrival_delay_minutes as delay_minutes,
                           j.food_satisfaction_score as satisfaction,
                           j.feedback_ID as feedback_id
                    ORDER BY j.feedback_ID DESC
                    LIMIT 10
                """,
                "on_time_flights": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    WHERE j.arrival_delay_minutes <= 15
                    WITH f, 
                         COUNT(j) as on_time_count,
                         COUNT(*) as total_count,
                         AVG(j.arrival_delay_minutes) as avg_delay
                    WHERE total_count >= 5
                    RETURN f.flight_number as flight_number,
                           on_time_count,
                           total_count,
                           (toFloat(on_time_count) / total_count * 100) as on_time_percentage,
                           avg_delay
                    ORDER BY on_time_percentage DESC
                    LIMIT 20
                """
            },
            "loyalty_analysis": {
                "loyalty_passenger_analysis": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    WHERE j.passenger_class IS NOT NULL
                    RETURN j.passenger_class as passenger_class,
                           COUNT(j) as journey_count,
                           AVG(j.food_satisfaction_score) as avg_satisfaction,
                           AVG(j.arrival_delay_minutes) as avg_delay,
                           AVG(j.actual_flown_miles) as avg_miles
                    ORDER BY journey_count DESC
                """,
                "loyalty_by_class": """
                    MATCH (j:Journey)
                    WHERE j.passenger_class IS NOT NULL
                    RETURN j.passenger_class as class,
                           COUNT(DISTINCT j.feedback_ID) as unique_passengers,
                           COUNT(j) as total_journeys,
                           AVG(j.food_satisfaction_score) as avg_satisfaction
                    ORDER BY total_journeys DESC
                """
            }
        }
    
    def retrieve(self, intent: str, entities: dict) -> tuple:
        """
        Retrieve information based on intent and entities.
        
        Args:
            intent: Classified intent from preprocessing layer
            entities: Extracted entities from preprocessing layer
                     Format: {"ENTITY_TYPE": [{"value": ..., "type": "ENTITY_TYPE"}, ...], ...}
                     Entity types: AIRPORT, FLIGHT, PASSENGER, JOURNEY, ROUTE, DATE, NUMBER
            
        Returns:
            Tuple of (list of retrieved records, list of executed queries with info)
        """
        # Input validation
        if not isinstance(intent, str):
            logger.warning(f"Invalid intent type: {type(intent)}, defaulting to 'general_question'")
            intent = "general_question"
        
        if not isinstance(entities, dict):
            logger.warning(f"Invalid entities type: {type(entities)}, using empty dict")
            entities = {}
        
        if intent not in self.query_templates:
            logger.info(f"Intent '{intent}' not in templates, defaulting to 'general_question'")
            intent = "general_question"
        
        templates = self.query_templates[intent]
        results = []
        executed_queries = []
        
        # For flight_search, prioritize by_route when both airports are present
        if intent == "flight_search" and entities.get("AIRPORT"):
            airports = entities.get("AIRPORT", [])
            if not isinstance(airports, list):
                airports = []
            airport_codes = [a["value"] for a in airports if isinstance(a, dict) and a.get("type") == "AIRPORT_CODE"]
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
                else:
                    logger.debug(f"Skipping template '{template_name}' - required parameters missing")
            except Exception as e:
                logger.error(f"Error executing query {template_name}: {e}", exc_info=True)
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
        """
        Build query parameters from extracted entities.
        
        Args:
            entities: Dictionary of extracted entities from preprocessing layer
                     Format: {"ENTITY_TYPE": [{"value": ..., "type": "ENTITY_TYPE"}, ...], ...}
            template_name: Name of the query template being used
            
        Returns:
            Dictionary of parameters for Cypher query, or None if required parameters are missing
        """
        if not isinstance(entities, dict):
            return None
        
        if not isinstance(template_name, str):
            return None
        
        parameters = {}
        
        # Templates that don't require parameters (can run without entities)
        # Check this FIRST before processing entities to avoid false matches
        no_param_templates = [
            "overall_statistics", 
            "popular_routes", 
            "satisfaction_by_class",
            "delays_by_route",
            "worst_delayed_flights",
            "loyalty_passenger_journeys",
            "multi_leg_journeys",
            "loyalty_passenger_analysis",
            "loyalty_by_class",
            "popular_flights",
            "cancelled_flight_patterns",
            "flight_reliability",
            "class_performance",
            "baggage_related_journeys",
            "on_time_flights",
            "best_routes"
        ]
        
        # If no specific parameters needed, return empty dict early
        if template_name in no_param_templates:
            return {}
        
        # Extract airport codes (consistent format: list of dicts with "value" and "type" keys)
        airports = entities.get("AIRPORT", [])
        if not isinstance(airports, list):
            airports = []
        
        airport_codes = []
        airport_names = []
        for a in airports:
            if isinstance(a, dict) and "value" in a:
                if a.get("type") == "AIRPORT_CODE":
                    airport_codes.append(a["value"])
                elif a.get("type") == "AIRPORT_NAME":
                    airport_names.append(a["value"])
        
        # Map airport names to codes (simplified - would need a mapping)
        # Prioritize by_route when both airports are present
        # Note: Check for exact template names, not substring matches
        if template_name == "by_route" or (template_name.startswith("by_") and "route" in template_name):
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
        
        # Extract flight numbers (consistent format: list of dicts with "value" and "type" keys)
        flights = entities.get("FLIGHT", [])
        if not isinstance(flights, list):
            flights = []
        
        if flights and isinstance(flights[0], dict) and "value" in flights[0]:
            parameters["flight_number"] = flights[0]["value"]
        
        # Extract numbers for thresholds (consistent format: list of dicts with "value" and "type" keys)
        numbers = entities.get("NUMBER", [])
        if not isinstance(numbers, list):
            numbers = []
        
        if numbers and isinstance(numbers[0], dict) and "value" in numbers[0]:
            try:
                if "delay" in template_name:
                    parameters["min_delay"] = float(numbers[0]["value"])
                elif "satisfaction" in template_name or "score" in template_name:
                    parameters["min_score"] = float(numbers[0]["value"])
            except (ValueError, TypeError):
                # Invalid number value, skip
                pass
        else:
            # Provide default thresholds if no number specified
            if "delay" in template_name and "flights_with_delays" in template_name:
                # Default: delays over 15 minutes
                parameters["min_delay"] = 15.0
            elif "satisfaction" in template_name and "low_rated_journeys" in template_name:
                # Default: satisfaction below 3
                parameters["min_score"] = 3.0
        
        # Extract Journey IDs (for journey_details query)
        # Consistent format: list of dicts with "value" and "type" keys
        journeys = entities.get("JOURNEY", [])
        if not isinstance(journeys, list):
            journeys = []
        
        if journeys and isinstance(journeys[0], dict) and "value" in journeys[0]:
            journey_id = str(journeys[0]["value"])
            # Normalize journey ID format (handle different formats like journey_12345, J12345, etc.)
            # Entity extractor returns just the numeric ID, but Neo4j stores it as feedback_ID
            # Try to match the format used in the database
            if journey_id.isdigit():
                # If it's just numbers, use as-is (Neo4j feedback_ID might be numeric or string)
                parameters["feedback_id"] = journey_id
            elif not journey_id.startswith(("journey_", "J", "j")):
                # If it doesn't have a prefix, try adding one
                parameters["feedback_id"] = f"journey_{journey_id}"
            else:
                parameters["feedback_id"] = journey_id
        
        # Extract Passenger IDs (for future passenger-specific queries)
        # Consistent format: list of dicts with "value" and "type" keys
        passengers = entities.get("PASSENGER", [])
        if not isinstance(passengers, list):
            passengers = []
        
        if passengers and isinstance(passengers[0], dict) and "value" in passengers[0]:
            passenger_id = str(passengers[0]["value"])
            # Normalize passenger ID format
            if passenger_id.isdigit():
                parameters["passenger_id"] = passenger_id
            elif not passenger_id.startswith(("passenger_", "P", "p")):
                parameters["passenger_id"] = f"passenger_{passenger_id}"
            else:
                parameters["passenger_id"] = passenger_id
        
        # Extract Dates (for time-based filtering)
        # Consistent format: list of dicts with "value" and "type" keys
        dates = entities.get("DATE", [])
        if not isinstance(dates, list):
            dates = []
        
        if dates and isinstance(dates[0], dict) and "value" in dates[0]:
            date_value = dates[0]["value"]
            parameters["date"] = str(date_value)
        
        # Extract Route mentions (routes are usually implicit via airport pairs, but can be explicit)
        # Consistent format: list of dicts with "value" and "type" keys
        routes = entities.get("ROUTE", [])
        if not isinstance(routes, list):
            routes = []
        
        if routes and isinstance(routes[0], dict) and "value" in routes[0]:
            route_value = routes[0]["value"]
            if route_value != "mentioned":  # If specific route name provided
                parameters["route_name"] = str(route_value)
        
        
        # For queries that require parameters, return None if missing
        # This allows queries without required params to be skipped
        return parameters if parameters else None

