"""Baseline retrieval using Cypher queries."""
from utils.neo4j_connector import Neo4jConnector
import logging
import re

logger = logging.getLogger(__name__)
import config


class BaselineRetriever:
    """Retrieves information from Neo4j using Cypher queries."""
    
    # Class-level constants
    NO_PARAM_TEMPLATES = {
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
        "best_routes",
        "general_flight_performance"
    }
    
    REQUIRED_PARAMS = {
        "by_route": {"departure_code", "arrival_code"},
        "by_departure": {"departure_code"},
        "by_arrival": {"arrival_code"},
        "flights_with_delays": {"min_delay"},
        "low_rated_journeys": {"min_score"},
        "journey_details": {"feedback_id"},
        "flight_info": {"flight_number"},
        "flight_info_with_passengers": {"flight_number"},
        "flight_performance_by_number": {"flight_number"},
        "flight_performance_recent": {"flight_number"},
    }
    
    # Default threshold values
    DEFAULT_MIN_DELAY = 15.0
    DEFAULT_MIN_SCORE = 3.0
    
    def __init__(self, connector: Neo4jConnector):
        self.connector = connector
        self.query_templates = self._initialize_query_templates()
        self._airport_cache = None  # Cache for valid airport codes
    
    def _get_valid_airports(self) -> set:
        """Get set of valid airport codes from the database."""
        if self._airport_cache is not None:
            return self._airport_cache
        
        try:
            query = "MATCH (a:Airport) RETURN DISTINCT a.station_code as code"
            result = self.connector.execute_query(query, {})
            codes = {record.get("code") for record in result if record.get("code")}
            self._airport_cache = codes
            logger.info(f"Loaded {len(codes)} valid airport codes from database")
            return codes
        except Exception as e:
            logger.warning(f"Could not load airport codes from database: {e}")
            return set()
    
    def _validate_airport_codes(self, airport_codes: list) -> tuple:
        """
        Validate airport codes against the database.
        
        Returns:
            Tuple of (valid_codes, invalid_codes)
        """
        if not airport_codes:
            return [], []
        
        valid_airports = self._get_valid_airports()
        valid = []
        invalid = []
        
        for code in airport_codes:
            if code in valid_airports:
                valid.append(code)
            else:
                invalid.append(code)
        
        return valid, invalid
    
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
                    MATCH (f:Flight)-[:DEPARTS_FROM]->(dep:Airport),
                          (f)-[:ARRIVES_AT]->(arr:Airport)
                    WHERE dep.station_code = $departure_code
                    RETURN f.flight_number as flight_number,
                           f.fleet_type_description as fleet_type,
                           dep.station_code as departure_airport,
                           arr.station_code as arrival_airport
                    LIMIT 20
                """,
                "by_arrival": """
                    MATCH (f:Flight)-[:ARRIVES_AT]->(arr:Airport),
                          (f)-[:DEPARTS_FROM]->(dep:Airport)
                    WHERE arr.station_code = $arrival_code
                    RETURN f.flight_number as flight_number,
                           f.fleet_type_description as fleet_type,
                           dep.station_code as departure_airport,
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
                "general_flight_performance": """
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
                """,
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
            airport_codes = []
            for a in airports:
                if not isinstance(a, dict) or "value" not in a:
                    continue
                t = a.get("type")
                # Normalize: keep only A-Z characters, then upper
                val_raw = str(a["value"])
                val_clean = re.sub(r'[^A-Za-z]', '', val_raw).upper()
                if t in {"AIRPORT_CODE", "AIRPORT"} and len(val_clean) == 3:
                    airport_codes.append(val_clean)
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
        
        # Validate airport codes for flight_search queries (cache validation results)
        invalid_airport_codes = set()
        if intent == "flight_search" and entities.get("AIRPORT"):
            airports = entities.get("AIRPORT", [])
            if isinstance(airports, list):
                airport_codes = []
                for a in airports:
                    if isinstance(a, dict) and "value" in a:
                        t = a.get("type")
                        val_raw = str(a["value"])
                        val_clean = re.sub(r'[^A-Za-z]', '', val_raw).upper()
                        if t in {"AIRPORT_CODE", "AIRPORT"} and len(val_clean) == 3:
                            airport_codes.append(val_clean)
                
                if airport_codes:
                    valid_codes, invalid_codes = self._validate_airport_codes(airport_codes)
                    if invalid_codes:
                        invalid_airport_codes = set(invalid_codes)
                        logger.warning(f"Invalid airport codes: {invalid_codes}")
        
        # Try different query templates based on available entities
        for template_name in template_order:
            if template_name not in templates:
                continue
            query = templates[template_name]
            try:
                parameters = self._build_parameters(entities, template_name, intent)
                if parameters is not None:
                    result = self.connector.execute_query(query, parameters)
                    
                    # Check if this specific query uses invalid airport codes
                    query_warnings = []
                    if invalid_airport_codes and intent == "flight_search":
                        # Check which airport codes are used in this query's parameters
                        codes_used = []
                        if "departure_code" in parameters:
                            codes_used.append(parameters["departure_code"])
                        if "arrival_code" in parameters:
                            codes_used.append(parameters["arrival_code"])
                        
                        # Only warn if this query uses invalid codes
                        invalid_used = [code for code in codes_used if code in invalid_airport_codes]
                        if invalid_used:
                            query_warnings.append(f"Airport code(s) not found in database: {', '.join(invalid_used)}")
                    
                    # Track executed query even if no results, include warnings
                    query_info = {
                        "template": template_name,
                        "intent": intent,
                        "query": query.strip(),
                        "parameters": parameters,
                        "result_count": len(result) if result else 0
                    }
                    if query_warnings:
                        query_info["warnings"] = query_warnings
                    executed_queries.append(query_info)
                    if result:
                        results.extend(result)
                else:
                    logger.debug(f"Skipping template '{template_name}' - required parameters missing")
            except Exception as e:
                logger.error(f"Error executing query {template_name}: {e}", exc_info=True)
                continue
        
        # Remove duplicates using stable keys
        seen = set()
        unique_results = []
        for record in results:
            # Use stable key: flight_number > feedback_id > full dict stringify
            if isinstance(record, dict):
                if "flight_number" in record:
                    key = f"flight_{record['flight_number']}"
                elif "feedback_id" in record:
                    key = f"journey_{record['feedback_id']}"
                else:
                    key = str(sorted(record.items()))
            else:
                key = str(record)
            
            if key not in seen:
                seen.add(key)
                unique_results.append(record)
        
        return unique_results[:50], executed_queries  # Limit total results
    
    def _build_parameters(self, entities: dict, template_name: str, intent: str = None) -> dict:
        """
        Build query parameters from extracted entities.
        
        Args:
            entities: Dictionary of extracted entities from preprocessing layer
                     Format: {"ENTITY_TYPE": [{"value": ..., "type": "ENTITY_TYPE"}, ...], ...}
            template_name: Name of the query template being used
            intent: Intent classification (optional, used for better flight number fallback)
            
        Returns:
            Dictionary of parameters for Cypher query, or None if required parameters are missing
        """
        if not isinstance(entities, dict):
            return None
        
        if not isinstance(template_name, str):
            return None
        
        parameters = {}
        
        # If no specific parameters needed, return empty dict early
        if template_name in self.NO_PARAM_TEMPLATES:
            return {}
        
        # Extract numbers
        numbers = entities.get("NUMBER", [])
        if not isinstance(numbers, list):
            numbers = []
        
        number_value = None
        if numbers and isinstance(numbers[0], dict) and "value" in numbers[0]:
            try:
                number_value = float(numbers[0]["value"])
            except (ValueError, TypeError):
                number_value = None
        
        # Delay threshold templates
        if template_name in {"flights_with_delays"}:
            parameters["min_delay"] = number_value if number_value is not None else self.DEFAULT_MIN_DELAY
        
        # Satisfaction threshold templates
        if template_name in {"low_rated_journeys"}:
            parameters["min_score"] = number_value if number_value is not None else self.DEFAULT_MIN_SCORE
        
        # Extract airport codes (consistent format: list of dicts with "value" and "type" keys)
        # Normalize with regex: strip non-letters, upper, then length check
        airports = entities.get("AIRPORT", [])
        if not isinstance(airports, list):
            airports = []
        
        airport_codes = []
        airport_names = []
        for a in airports:
            if not isinstance(a, dict) or "value" not in a:
                continue
            t = a.get("type")
            # Normalize: keep only A-Z characters, then upper
            val_raw = str(a["value"])
            val_clean = re.sub(r'[^A-Za-z]', '', val_raw).upper()
            if t in {"AIRPORT_CODE", "AIRPORT"} and len(val_clean) == 3:
                airport_codes.append(val_clean)
            elif t == "AIRPORT_NAME":
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
        # Fallback to NUMBER if FLIGHT is missing, but only if safe
        if template_name in {"flight_info", "flight_info_with_passengers",
                             "flight_performance_by_number", "flight_performance_recent"}:
            if "flight_number" not in parameters:
                flights = entities.get("FLIGHT", [])
                if isinstance(flights, list) and flights and isinstance(flights[0], dict):
                    parameters["flight_number"] = str(flights[0].get("value"))
            
            # Fallback to NUMBER only if:
            # 1. FLIGHT entity exists (prefer it) OR intent is flight-related
            # 2. Number is >= 1 and int-like
            # 3. Avoid small threshold values (3, 4, 15) that might be satisfaction/delay thresholds
            if "flight_number" not in parameters:
                has_flight_entity = bool(entities.get("FLIGHT"))
                is_flight_intent = intent in {"general_question", "flight_performance", "performance_metrics"}
                
                if has_flight_entity or is_flight_intent:
                    numbers = entities.get("NUMBER", [])
                    if isinstance(numbers, list) and numbers and isinstance(numbers[0], dict):
                        v = numbers[0].get("value")
                        if v is not None:
                            try:
                                num_val = float(v)
                                # Only use if >= 1, int-like, and not a common threshold value
                                if num_val >= 1 and num_val == int(num_val) and num_val not in {3, 4, 15}:
                                    parameters["flight_number"] = str(int(num_val))
                            except (ValueError, TypeError):
                                pass
        
        
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
        
        
        # Validate required parameters
        required = self.REQUIRED_PARAMS.get(template_name, set())
        if required and not required.issubset(parameters.keys()):
            return None
        
        return parameters

