"""Structured prompt templates for LLM interactions."""


def build_prompt(context: str, persona: str, task: str, user_query: str) -> str:
    """
    Build a structured prompt with context, persona, and task.
    
    Args:
        context: Retrieved knowledge graph information
        persona: Assistant's role definition
        task: Instructions for what to do
        user_query: Original user query
        
    Returns:
        Complete prompt string
    """
    prompt = f"""{persona}

{task}

Context from Knowledge Graph:
{context}

User Query: {user_query}

Answer:"""
    
    return prompt


def get_persona() -> str:
    """Get the persona definition for the airline assistant."""
    return """You are an Airline Company Flight Insights Assistant. Your role is to help airline management understand flight performance, passenger satisfaction, delays, and operational insights. You provide data-driven insights based on the knowledge graph information provided."""


def get_task_instruction() -> str:
    """Get the task instruction for the assistant."""
    return """Based on the context provided from the knowledge graph, answer the user's question accurately and comprehensively. Use only the information provided in the context. If the context doesn't contain enough information to answer the question, say so. Be specific with numbers, flight numbers, airport codes, and metrics when available."""

