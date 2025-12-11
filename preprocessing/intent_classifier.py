"""Intent classification for routing queries to appropriate retrieval strategies."""
import re
import logging
import config

# Configure logging
logger = logging.getLogger(__name__)


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
                r"low.*rating"
            ],
            "route_analysis": [
                r"route",
                r"path",
                r"connection",
                r"stop",
                r"direct",
                r"indirect"
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
                r"compare"
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
            ]
        }
    
    def classify(self, query: str) -> str:
        """
        Classify the user query into an intent category.
        
        Args:
            query: User input query
            
        Returns:
            Intent category string (defaults to "general_question" if classification fails)
            
        Raises:
            ValueError: If query is None or empty
        """
        if query is None:
            logger.warning("None query provided, defaulting to general_question")
            return "general_question"
        
        if not isinstance(query, str):
            logger.warning(f"Non-string query provided ({type(query)}), defaulting to general_question")
            return "general_question"
        
        if not query.strip():
            logger.warning("Empty query provided, defaulting to general_question")
            return "general_question"
        
        try:
            query_lower = query.lower()
            
            # Score each intent
            intent_scores = {}
            for intent, patterns in self.intent_patterns.items():
                try:
                    score = sum(1 for pattern in patterns if re.search(pattern, query_lower, re.IGNORECASE))
                    if score > 0:
                        intent_scores[intent] = score
                except re.error as e:
                    logger.warning(f"Invalid regex pattern for intent '{intent}': {e}")
                    continue
            
            # Return the intent with highest score, or default to general_question
            if intent_scores:
                return max(intent_scores, key=intent_scores.get)
            return "general_question"
        except Exception as e:
            logger.error(f"Unexpected error during classification: {e}")
            return "general_question"
    
    def classify_with_llm(self, query: str, llm=None) -> str:
        """
        Classify intent using an LLM (optional enhancement).
        
        Args:
            query: User input query
            llm: Optional LLM instance for classification
            
        Returns:
            Intent category string (falls back to rule-based classification on error)
        """
        # Validate input
        if query is None or not isinstance(query, str) or not query.strip():
            logger.warning("Invalid query for LLM classification, using rule-based")
            return self.classify(query)
        
        if llm is None:
            logger.debug("No LLM provided, using rule-based classification")
            return self.classify(query)
        
        prompt = f"""Classify the following airline-related query into one of these intents:
{', '.join(config.INTENTS)}

Query: {query}

Respond with only the intent name."""
        
        try:
            response = llm.invoke(prompt)
            if response:
                intent = response.strip().lower()
                if intent in config.INTENTS:
                    logger.debug(f"LLM classified intent: {intent}")
                    return intent
                else:
                    logger.warning(f"LLM returned invalid intent '{intent}', falling back to rule-based")
            else:
                logger.warning("LLM returned empty response, falling back to rule-based")
        except AttributeError as e:
            logger.error(f"LLM object missing required method 'invoke': {e}")
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
        
        return self.classify(query)

