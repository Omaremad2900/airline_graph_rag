"""Entity extraction from user queries using NER."""
import re
import config


class EntityExtractor:
    """Extracts entities from user queries for airline domain."""
    
    def __init__(self):
        # Airport codes pattern (3-letter IATA codes)
        self.airport_code_pattern = r'\b[A-Z]{3}\b'
        
        # Common airport names (can be extended)
        self.airport_names = {
            "jfk", "lax", "ord", "dfw", "atl", "den", "sfo", "las", "mco",
            "phx", "mia", "sea", "iah", "ewr", "msp", "dtw", "bwi", "iad",
            "cairo", "london", "paris", "dubai", "tokyo", "beijing", "new york",
            "los angeles", "chicago", "atlanta", "dallas", "miami"
        }
        
        # Flight number pattern (e.g., AA123, DL456)
        self.flight_number_pattern = r'\b[A-Z]{2}\d{3,4}\b'
        
        # Date patterns
        self.date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{4}\b'  # Year only
        ]
        
        # Number patterns
        self.number_pattern = r'\b\d+(?:\.\d+)?\b'
    
    def extract_airports(self, query: str) -> list:
        """Extract airport codes and names from query."""
        airports = []
        query_upper = query.upper()
        
        # Extract airport codes
        codes = re.findall(self.airport_code_pattern, query_upper)
        airports.extend([{"value": code, "type": "AIRPORT_CODE"} for code in codes])
        
        # Extract airport names
        query_lower = query.lower()
        for airport in self.airport_names:
            if airport in query_lower:
                airports.append({"value": airport, "type": "AIRPORT_NAME"})
        
        return airports
    
    def extract_flights(self, query: str) -> list:
        """Extract flight numbers from query."""
        flights = re.findall(self.flight_number_pattern, query.upper())
        return [{"value": flight, "type": "FLIGHT"} for flight in flights]
    
    def extract_dates(self, query: str) -> list:
        """Extract dates from query."""
        dates = []
        for pattern in self.date_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            dates.extend([{"value": match, "type": "DATE"} for match in matches])
        return dates
    
    def extract_numbers(self, query: str) -> list:
        """Extract numeric values from query."""
        numbers = re.findall(self.number_pattern, query)
        return [{"value": float(num), "type": "NUMBER"} for num in numbers]
    
    def extract_entities(self, query: str) -> dict:
        """
        Extract all entities from the query.
        
        Args:
            query: User input query
            
        Returns:
            Dictionary of extracted entities by type
        """
        entities = {
            "AIRPORT": self.extract_airports(query),
            "FLIGHT": self.extract_flights(query),
            "DATE": self.extract_dates(query),
            "NUMBER": self.extract_numbers(query)
        }
        
        # Filter out empty lists
        return {k: v for k, v in entities.items() if v}
    
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

