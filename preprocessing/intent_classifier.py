"""Intent classification for routing queries to appropriate retrieval strategies."""
import re


class IntentClassifier:
    """Classifies user queries into intent categories."""
    
    def __init__(self):
        self.intent_patterns = {
            "flight_search": [
                r"find.*flight",
                r"search.*flight",
                r"flights?.*from.*to",
                r"flights?.*between",
                r"available.*flights?"
            ],
            "delay_analysis": [
                r"delay",
                r"late",
                r"on.?time",
                r"arrival.*delay",
                r"departure.*delay",
                r"delayed.*flights?"
            ],
            "passenger_satisfaction": [
                r"satisfaction",
                r"rating",
                r"review",
                r"feedback",
                r"satisfied",
                r"complaint",
                r"food.*satisfaction",
                r"poor.*rating",
                r"low.*rating",
                r"low.*rated",
                r"rated.*journey"
            ],
            "route_analysis": [
                r"route",
                r"path",
                r"connection",
                r"stop",
                r"direct",
                r"indirect",
                r"multi.*leg",
                r"multi-leg",
                r"multi.*leg.*journey"
            ],
            "journey_insights": [
                r"journey",
                r"trip",
                r"travel",
                r"passenger.*journey",
                r"journey.*details"
            ],
            "performance_metrics": [
                r"performance",
                r"metric",
                r"statistic",
                r"average",
                r"top",
                r"worst",
                r"best",
                r"compare",
                r"overall.*statistic"
            ],
            "recommendation": [
                r"recommend",
                r"suggest",
                r"best.*option",
                r"should.*choose",
                r"which.*better"
            ],
            "general_question": [
                r"what",
                r"how",
                r"why",
                r"when",
                r"where",
                r"tell.*me",
                r"explain"
            ],
            "flight_performance": [
                r"flight.*status",
                r"status.*of.*flight",
                r"flight.*performance",
                r"flight.*metrics",
                r"flight.*analysis",
                r"how.*is.*flight.*performing",
                r"flight.*on.*time",
                r"flight.*reliability",
                r"performance.*of.*flight",
                r"flight.*status.*analysis",
                r"flight.*reliability.*metric",
                r"what.*is.*the.*performance.*of.*flight",
                r"what.*is.*performance.*of.*flight"
            ],
            "loyalty_analysis": [
                r"frequent.*flyer",
                r"loyalty.*program",
                r"loyalty.*analysis",
                r"miles.*analysis",
                r"points.*analysis",
                r"loyalty.*metrics",
                r"membership.*analysis",
                r"tier.*analysis",
                r"loyalty.*metric.*by.*class",
                r"loyalty.*by.*class"
            ]
        }
    
    def classify(self, query: str) -> str:
        """
        Classify the user query into an intent category.
        
        Args:
            query: User input query
            
        Returns:
            Intent category string
        """
        query_lower = query.lower()
        
        # Score each intent
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query_lower, re.IGNORECASE))
            if score > 0:
                intent_scores[intent] = score
        
        # Context-aware disambiguation for ambiguous queries
        if len(intent_scores) > 1:
            # Priority 0: Recommendation intent should take precedence when "recommend" or "suggest" is present
            if "recommendation" in intent_scores:
                intent_scores["recommendation"] += 3  # Boost recommendation significantly
                # Reduce competing intents
                if "performance_metrics" in intent_scores:
                    intent_scores["performance_metrics"] = max(0, intent_scores["performance_metrics"] - 1)
                if "route_analysis" in intent_scores:
                    intent_scores["route_analysis"] = max(0, intent_scores["route_analysis"] - 1)
            
            # Priority 0.5: Flight performance should take precedence over performance_metrics when "flight" is mentioned
            if "flight_performance" in intent_scores:
                # If query mentions "flight" with performance-related terms, prefer flight_performance
                if "performance_metrics" in intent_scores and "flight" in query_lower:
                    # Check if it's asking about a specific flight (has flight number pattern or "what is")
                    has_flight_number = bool(re.search(r'\b[A-Z]{2}\d{3,4}\b', query_lower))
                    has_what_is = "what is" in query_lower or "how is" in query_lower
                    has_performance_of = "performance of" in query_lower or "performance of flight" in query_lower
                    has_reliability = "reliability" in query_lower
                    has_status = "status" in query_lower
                    # If it has flight + performance keywords, prefer flight_performance
                    if has_flight_number or has_what_is or has_performance_of or has_reliability or has_status:
                        intent_scores["flight_performance"] += 5  # Boost flight_performance significantly for flight-related performance queries
                        intent_scores["performance_metrics"] = max(0, intent_scores["performance_metrics"] - 2)  # Reduce performance_metrics
                    else:
                        intent_scores["performance_metrics"] += 2  # Otherwise prefer performance_metrics
                if "flight_search" in intent_scores:
                    # If query mentions performance, status, analysis, or reliability, prefer flight_performance
                    if any(word in query_lower for word in ["performance", "status", "analysis", "reliability", "metric"]):
                        intent_scores["flight_performance"] += 5  # Boost flight_performance significantly over flight_search
                        intent_scores["flight_search"] = max(0, intent_scores["flight_search"] - 2)  # Reduce flight_search significantly
                    else:
                        intent_scores["flight_search"] = max(0, intent_scores["flight_search"] - 1)  # Reduce flight_search
                if "delay_analysis" in intent_scores:
                    intent_scores["flight_performance"] += 2  # Boost flight_performance over delay_analysis
                if "general_question" in intent_scores:
                    intent_scores["flight_performance"] += 2  # Boost flight_performance over general_question
            
            # Priority 0.6: Loyalty analysis should take precedence over performance_metrics when "loyalty" is mentioned
            if "loyalty_analysis" in intent_scores:
                if "performance_metrics" in intent_scores and "loyalty" in query_lower:
                    intent_scores["loyalty_analysis"] += 3  # Boost loyalty_analysis over performance_metrics
            
            # Priority 0.7: Passenger satisfaction should take precedence over journey_insights when "rated" or "rating" is mentioned
            if "passenger_satisfaction" in intent_scores:
                if "journey_insights" in intent_scores and ("rated" in query_lower or "rating" in query_lower):
                    intent_scores["passenger_satisfaction"] += 3  # Boost passenger_satisfaction over journey_insights
            
            # Priority 0.8: Route analysis should take precedence over journey_insights when "multi-leg" is mentioned
            if "route_analysis" in intent_scores:
                if "journey_insights" in intent_scores and ("multi" in query_lower or "leg" in query_lower):
                    intent_scores["route_analysis"] += 4  # Boost route_analysis over journey_insights significantly
            
            # Priority 2: If "flight" appears and flight_search is a candidate, boost it over general_question
            if "flight" in query_lower and "flight_search" in intent_scores:
                if "general_question" in intent_scores:
                    intent_scores["flight_search"] += 2  # Boost flight_search
            
            # If query has action verbs with specific entities, prefer specific intent
            action_verbs = ["find", "show", "get", "search", "list", "display", "analyze"]
            if any(verb in query_lower for verb in action_verbs):
                # Prefer specific intents over general_question
                if "general_question" in intent_scores and len(intent_scores) > 1:
                    # Reduce general_question score if other specific intents exist
                    intent_scores["general_question"] = max(0, intent_scores["general_question"] - 1)
            
            # Special case: "compare" should prefer performance_metrics over flight_search and flight_performance
            if "compare" in query_lower:
                if "performance_metrics" in intent_scores:
                    intent_scores["performance_metrics"] += 3  # Boost performance_metrics significantly
                # Reduce competing intents
                if "flight_search" in intent_scores:
                    intent_scores["flight_search"] = max(0, intent_scores["flight_search"] - 1)
                if "flight_performance" in intent_scores:
                    intent_scores["flight_performance"] = max(0, intent_scores["flight_performance"] - 1)
        
        # Additional heuristic: if query is short and contains "flight", prefer flight_search
        # BUT only if it doesn't contain performance-related keywords
        if not intent_scores or (len(query_lower.split()) <= 3 and "flight" in query_lower):
            if "flight" in query_lower:
                # Don't add flight_search if query has performance/status/analysis keywords
                performance_keywords = ["performance", "status", "analysis", "reliability", "metric"]
                if not any(keyword in query_lower for keyword in performance_keywords):
                    # Add flight_search if not already present
                    if "flight_search" not in intent_scores:
                        intent_scores["flight_search"] = 1
                    # Boost it significantly
                    intent_scores["flight_search"] += 3
        
        # Return the intent with highest score, or default to general_question
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        return "general_question"

