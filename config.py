"""Configuration settings for the Graph-RAG system."""
import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Embedding Models (for comparison)
EMBEDDING_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "name": "MiniLM-L6-v2",
        "dimension": 384,
        "type": "sentence-transformers"
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "name": "MPNet-Base-v2",
        "dimension": 768,
        "type": "sentence-transformers"
    }
}

# LLM Models Configuration
LLM_MODELS = {
    "gpt-3.5-turbo": {
        "provider": "openai",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    "gpt-4": {
        "provider": "openai",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    "claude-3-haiku-20240307": {
        "provider": "anthropic",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    "mistralai/mistral-7b-instruct": {
        "provider": "openrouter",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    "meta-llama/llama-3.1-8b-instruct": {
        "provider": "openrouter",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    "llama-3.3-70b-versatile": {
        "provider": "groq",
        "max_tokens": 1000,
        "temperature": 0.7
    }
}

# Intent Classification
INTENTS = [
    "flight_search",
    "delay_analysis",
    "passenger_satisfaction",
    "route_analysis",
    "journey_insights",
    "performance_metrics",
    "recommendation",
    "general_question",
    "booking",
    "cancellation",
    "check_in",
    "flight_status",
    "seat_selection",
    "baggage",
    "loyalty"
]

# Entity Types
ENTITY_TYPES = [
    "AIRPORT",
    "FLIGHT",
    "PASSENGER",
    "JOURNEY",
    "ROUTE",
    "DATE",
    "NUMBER"
]

