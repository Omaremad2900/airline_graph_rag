"""Entity extraction from user queries using NER."""
import re
import logging
import config

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts entities from user queries for airline domain."""
    
    def __init__(self):
        # Valid IATA airport codes (whitelist to prevent false positives)
        self.valid_airport_codes = {
            # Major US airports
            "JFK", "LAX", "ORD", "DFW", "ATL", "DEN", "SFO", "LAS", "MCO",
            "PHX", "MIA", "SEA", "IAH", "EWR", "MSP", "DTW", "BWI", "IAD",
            "BOS", "CLT", "LGA", "DCA", "SLC", "PDX", "BNA", "AUS", "SJC",
            "OAK", "RDU", "MCI", "STL", "TPA", "SAN", "DAL", "HOU", "FLL",
            # International airports
            "LHR", "CDG", "AMS", "FRA", "DXB", "NRT", "HND", "PEK", "PVG",
            "CAN", "SIN", "ICN", "BKK", "KUL", "IST", "MAD", "BCN", "FCO",
            "MUC", "ZUR", "VIE", "CPH", "OSL", "ARN", "HEL", "DUB", "MAN",
            "YYZ", "YVR", "YUL", "SYD", "MEL", "AKL", "JNB", "CAI", "DXB"
        }
        
        # Common English words to exclude (3-letter uppercase words that aren't airports)
        self.excluded_words = {
            "THE", "AND", "FOR", "ARE", "ALL", "NEW", "OLD", "ONE", "TWO",
            "SIX", "TEN", "DAY", "WAY", "MAY", "JAN", "FEB", "MAR", "APR",
            "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC", "YES", "NOT",
            "BUT", "HER", "HIS", "OUR", "OUT", "USE", "WHO", "WHY", "HOW",
            "WHEN", "WHAT", "WHERE", "FROM", "WITH", "INTO", "UPON", "OVER"
        }
        
        # Airport codes pattern (3-letter IATA codes) - will be filtered by whitelist
        self.airport_code_pattern = r'\b[A-Z]{3}\b'
        
        # Common airport names (can be extended)
        self.airport_names = {
            "jfk", "lax", "ord", "dfw", "atl", "den", "sfo", "las", "mco",
            "phx", "mia", "sea", "iah", "ewr", "msp", "dtw", "bwi", "iad",
            "cairo", "london", "paris", "dubai", "tokyo", "beijing", "new york",
            "los angeles", "chicago", "atlanta", "dallas", "miami"
        }
        
        # Mapping of airport names to codes for deduplication
        self.airport_name_to_code = {
            "jfk": "JFK", "lax": "LAX", "ord": "ORD", "dfw": "DFW", "atl": "ATL",
            "den": "DEN", "sfo": "SFO", "las": "LAS", "mco": "MCO", "phx": "PHX",
            "mia": "MIA", "sea": "SEA", "iah": "IAH", "ewr": "EWR", "msp": "MSP",
            "dtw": "DTW", "bwi": "BWI", "iad": "IAD"
        }
        
        # Flight number pattern (e.g., AA123, DL456)
        self.flight_number_pattern = r'\b[A-Z]{2}\d{3,4}\b'
        
        # Journey ID pattern (e.g., journey_12345, journey-12345, J12345, j_12345)
        # Must have explicit prefix to avoid matching years or random numbers
        # Pattern: journey/j + optional separator + digits OR single letter + digits (no separator)
        self.journey_id_pattern = r'\b(?:(?:journey[_-]|j[_-])(\d{4,})|j(\d{4,}))\b'
        
        # Passenger ID pattern (e.g., passenger_12345, passenger-12345, P12345, p_12345)
        # Must have explicit prefix to avoid matching years or random numbers
        # Pattern: passenger/p + optional separator + digits OR single letter + digits (no separator)
        self.passenger_id_pattern = r'\b(?:(?:passenger[_-]|p[_-])(\d{4,})|p(\d{4,}))\b'
        
        # Route pattern (implicitly via airport pairs, but can extract route mentions)
        self.route_pattern = r'\broute[_-]?(\w+)?\b'
        
        # Date patterns (ordered by specificity - most specific first)
        self.date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',  # Month name format
        ]
        
        # Year-only pattern (used only if not part of full date)
        self.year_pattern = r'\b\d{4}\b'
        
        # Number patterns
        self.number_pattern = r'\b\d+(?:\.\d+)?\b'
    
    def _normalize_airport(self, value: str, entity_type: str) -> str:
        """Normalize airport code/name for deduplication."""
        if entity_type == "AIRPORT_CODE":
            return value.upper()
        elif entity_type == "AIRPORT_NAME":
            # Check if name maps to a code
            normalized = value.lower()
            if normalized in self.airport_name_to_code:
                return self.airport_name_to_code[normalized]
            return normalized
        return value.lower()
    
    def extract_airports(self, query: str) -> list:
        """Extract airport codes and names from query."""
        if not query or not isinstance(query, str):
            return []
        
        airports = []
        query_upper = query.upper()
        query_lower = query.lower()
        
        # Extract airport codes (filtered by whitelist)
        codes = re.findall(self.airport_code_pattern, query_upper)
        for code in codes:
            # Only include if it's a valid airport code and not an excluded word
            if code in self.valid_airport_codes and code not in self.excluded_words:
                airports.append({"value": code, "type": "AIRPORT_CODE"})
        
        # Extract airport names
        for airport in self.airport_names:
            if airport in query_lower:
                airports.append({"value": airport, "type": "AIRPORT_NAME"})
        
        # Deduplicate: if same airport found as both code and name, keep only code
        seen = set()
        deduplicated = []
        for airport in airports:
            normalized = self._normalize_airport(airport["value"], airport["type"])
            if normalized not in seen:
                seen.add(normalized)
                deduplicated.append(airport)
            elif airport["type"] == "AIRPORT_CODE":
                # Replace name with code if we see the code version
                for i, existing in enumerate(deduplicated):
                    if self._normalize_airport(existing["value"], existing["type"]) == normalized:
                        if existing["type"] == "AIRPORT_NAME":
                            deduplicated[i] = airport  # Replace with code version
                            break
        
        return deduplicated
    
    def extract_flights(self, query: str) -> list:
        """Extract flight numbers from query."""
        if not query or not isinstance(query, str):
            return []
        
        flights = re.findall(self.flight_number_pattern, query.upper())
        return [{"value": flight, "type": "FLIGHT"} for flight in flights]
    
    def extract_journeys(self, query: str) -> list:
        """Extract journey IDs from query."""
        if not query or not isinstance(query, str):
            return []
        
        journeys = []
        # Extract journey IDs (e.g., journey_12345, journey-12345, J12345, j_12345)
        # Must have explicit prefix to avoid matching years or dates
        matches = re.finditer(self.journey_id_pattern, query, re.IGNORECASE)
        for match in matches:
            # Group 1: journey_12345 or j_12345 format
            # Group 2: J12345 format (single letter, no separator)
            journey_id = match.group(1) or match.group(2)
            if journey_id:
                journeys.append({"value": journey_id, "type": "JOURNEY"})
        return journeys
    
    def extract_passengers(self, query: str) -> list:
        """Extract passenger IDs from query."""
        if not query or not isinstance(query, str):
            return []
        
        passengers = []
        # Extract passenger IDs (e.g., passenger_12345, passenger-12345, P12345, p_12345)
        # Must have explicit prefix to avoid matching years or dates
        matches = re.finditer(self.passenger_id_pattern, query, re.IGNORECASE)
        for match in matches:
            # Group 1: passenger_12345 or p_12345 format
            # Group 2: P12345 format (single letter, no separator)
            passenger_id = match.group(1) or match.group(2)
            if passenger_id:
                passengers.append({"value": passenger_id, "type": "PASSENGER"})
        return passengers
    
    def extract_routes(self, query: str) -> list:
        """Extract route mentions from query (routes are typically implicit via airport pairs)."""
        if not query or not isinstance(query, str):
            return []
        
        # Routes are usually extracted as airport pairs, but we can identify route mentions
        routes = []
        matches = re.finditer(self.route_pattern, query, re.IGNORECASE)
        for match in matches:
            route_name = match.group(1)
            # If group 1 is None or empty, or if it's just a single letter (like 's' from 'routes'),
            # treat it as a route mention
            if not route_name or len(route_name) == 1:
                route_name = "mentioned"
            routes.append({"value": route_name, "type": "ROUTE"})
        return routes
    
    def extract_dates(self, query: str) -> list:
        """Extract dates from query."""
        if not query or not isinstance(query, str):
            return []
        
        dates = []
        full_date_matches = set()
        
        # Extract full dates first (most specific patterns)
        for pattern in self.date_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                dates.append({"value": match, "type": "DATE"})
                # Track the year in full dates to avoid extracting it separately
                year_match = re.search(r'\d{4}', match)
                if year_match:
                    full_date_matches.add(year_match.group())
        
        # Extract year-only if not part of a full date
        year_matches = re.findall(self.year_pattern, query)
        for year in year_matches:
            if year not in full_date_matches:
                dates.append({"value": year, "type": "DATE"})
        
        return dates
    
    def extract_numbers(self, query: str, exclude_dates: list = None, exclude_ids: set = None) -> list:
        """Extract numeric values from query, excluding those in dates and entity IDs."""
        if not query or not isinstance(query, str):
            return []
        
        if exclude_dates is None:
            exclude_dates = []
        if exclude_ids is None:
            exclude_ids = set()
        
        # Find all number positions
        number_matches = list(re.finditer(self.number_pattern, query))
        numbers = []
        
        # Find all date positions
        date_positions = []
        for date_entity in exclude_dates:
            date_value = date_entity["value"]
            for match in re.finditer(re.escape(date_value), query, re.IGNORECASE):
                date_positions.append((match.start(), match.end()))
        
        # Extract numbers that are not part of dates or entity IDs
        for match in number_matches:
            num_start, num_end = match.span()
            num_value = match.group()
            
            # Check if this number is part of any date
            is_in_date = False
            for date_start, date_end in date_positions:
                if date_start <= num_start < date_end or date_start < num_end <= date_end:
                    is_in_date = True
                    break
            
            # Check if this number is an entity ID (journey or passenger)
            if num_value in exclude_ids:
                continue
            
            if not is_in_date:
                numbers.append({"value": float(num_value), "type": "NUMBER"})
        
        return numbers
    
    def _map_qualitative_to_number(self, query: str) -> list:
        """
        Map qualitative terms to numeric thresholds for filtering.
        Handles terms like "low", "high", "poor", "excellent" related to satisfaction/ratings.
        
        Args:
            query: User input query (must be a non-empty string)
            
        Returns:
            List of number entities representing qualitative thresholds
            Multiple values can be returned if query contains multiple qualitative terms
            (e.g., "low satisfaction and high ratings" returns both 3.0 and 4.0)
        """
        if not query or not isinstance(query, str):
            return []
        
        numbers = []
        query_lower = query.lower()
        
        # Low satisfaction/ratings patterns (checked first, most specific patterns first)
        if re.search(r'\blow\s+(?:passenger\s+)?satisfaction\b', query_lower):
            numbers.append({"value": 3.0, "type": "NUMBER"})
        elif re.search(r'\blow\s+ratings?\b', query_lower):
            numbers.append({"value": 3.0, "type": "NUMBER"})
        elif re.search(r'\bpoor\s+(?:passenger\s+)?satisfaction\b', query_lower):
            numbers.append({"value": 2.5, "type": "NUMBER"})
        elif re.search(r'\bpoor\s+ratings?\b', query_lower):
            numbers.append({"value": 2.5, "type": "NUMBER"})
        
        # High satisfaction/ratings patterns (separate if to allow both low and high in same query)
        # Most specific patterns checked first
        if re.search(r'\bexcellent\s+(?:passenger\s+)?satisfaction\b', query_lower):
            numbers.append({"value": 4.5, "type": "NUMBER"})
        elif re.search(r'\bexcellent\s+ratings?\b', query_lower):
            numbers.append({"value": 4.5, "type": "NUMBER"})
        elif re.search(r'\bexcellent\s+feedback\b', query_lower):
            numbers.append({"value": 4.5, "type": "NUMBER"})
        elif re.search(r'\bhigh\s+(?:passenger\s+)?satisfaction\b', query_lower):
            numbers.append({"value": 4.0, "type": "NUMBER"})
        elif re.search(r'\bhigh\s+ratings?\b', query_lower):
            numbers.append({"value": 4.0, "type": "NUMBER"})
        
        return numbers
    
    def extract_entities(self, query: str) -> dict:
        """
        Extract all entities from the query.
        
        Args:
            query: User input query (must be a non-empty string)
            
        Returns:
            Dictionary of extracted entities by type (only includes non-empty entity lists)
            Format: {"ENTITY_TYPE": [{"value": ..., "type": "ENTITY_TYPE"}, ...], ...}
        """
        # Input validation
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided to extract_entities, returning empty dict")
            return {}
        
        if not query.strip():
            logger.warning("Empty query provided to extract_entities, returning empty dict")
            return {}
        
        try:
            # Extract dates first (needed to exclude numbers from dates)
            dates = self.extract_dates(query)
            
            # Extract journey and passenger IDs to exclude from number extraction
            # This prevents numbers that are part of entity IDs from being extracted as separate NUMBER entities
            journey_ids = set()
            passenger_ids = set()
            for journey in self.extract_journeys(query):
                journey_ids.add(journey["value"])
            for passenger in self.extract_passengers(query):
                passenger_ids.add(passenger["value"])
            exclude_ids = journey_ids | passenger_ids
            
            # Extract numeric values (excludes numbers in dates and entity IDs)
            numeric_values = self.extract_numbers(query, exclude_dates=dates, exclude_ids=exclude_ids)
            
            # Map qualitative terms to numbers (e.g., "low satisfaction" -> 3.0)
            # This extends the numeric_values list with qualitative mappings
            qualitative_numbers = self._map_qualitative_to_number(query)
            numeric_values.extend(qualitative_numbers)
            
            # Extract all entity types
            entities = {
                "AIRPORT": self.extract_airports(query),
                "FLIGHT": self.extract_flights(query),
                "PASSENGER": self.extract_passengers(query),
                "JOURNEY": self.extract_journeys(query),
                "ROUTE": self.extract_routes(query),
                "DATE": dates,
                "NUMBER": numeric_values
            }
            
            # Filter out empty lists to return only entities that were found
            result = {k: v for k, v in entities.items() if v}
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error during entity extraction: {e}", exc_info=True)
            return {}
    
    def extract_with_llm(self, query: str, llm=None) -> dict:
        """
        Extract entities using LLM (optional enhancement).
        
        Args:
            query: User input query
            llm: Optional LLM instance for extraction
            
        Returns:
            Dictionary of extracted entities
        """
        # Fallback to rule-based if no LLM
        if llm is None:
            return self.extract_entities(query)
        
        prompt = f"""Extract entities from this airline query. Return JSON with entity types and values:
Query: {query}

Entity types: AIRPORT, FLIGHT, PASSENGER, JOURNEY, DATE, NUMBER

Return format: {{"AIRPORT": ["code1", "code2"], "FLIGHT": ["AA123"], ...}}"""
        
        try:
            response = llm.invoke(prompt)
            # Parse JSON response (simplified - would need proper JSON parsing)
            # For now, fallback to rule-based
            return self.extract_entities(query)
        except Exception as e:
            print(f"LLM entity extraction failed: {e}")
            return self.extract_entities(query)

