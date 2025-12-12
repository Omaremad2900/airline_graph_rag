"""Evaluation utilities for comparing LLM models."""
import json
from typing import Dict, List
from llm_layer.models import LLMManager


class ModelEvaluator:
    """Evaluates and compares LLM models."""
    
    # Approximate token costs (USD per 1K tokens) - Update with current prices
    TOKEN_COSTS = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "gemini-pro": {"input": 0.0005, "output": 0.0015},
    }
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
    
    def estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for API call.
        
        Args:
            model_name: Name of the model
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        if model_name not in self.TOKEN_COSTS:
            return 0.0  # Unknown/free models
        
        costs = self.TOKEN_COSTS[model_name]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return input_cost + output_cost
    
    def evaluate_quantitative(self, results: Dict[str, Dict]) -> Dict:
        """
        Compute quantitative metrics for model comparison.
        
        Args:
            results: Dictionary of model responses and metrics
            
        Returns:
            Quantitative metrics dictionary
        """
        metrics = {}
        
        for model_name, data in results.items():
            model_metrics = data.get("metrics", {})
            response_length = len(data.get("response", ""))
            estimated_tokens = model_metrics.get("estimated_tokens", 0)
            
            # Estimate input/output tokens (rough approximation)
            input_tokens = estimated_tokens // 2
            output_tokens = estimated_tokens // 2
            
            metrics[model_name] = {
                "response_time": model_metrics.get("response_time", 0),
                "estimated_tokens": estimated_tokens,
                "response_length": response_length,
                "estimated_cost": self.estimate_cost(model_name, input_tokens, output_tokens),
                "words_per_second": (response_length / 4) / model_metrics.get("response_time", 1) if model_metrics.get("response_time", 0) > 0 else 0
            }
        
        return metrics
    
    def evaluate_qualitative(self, results: Dict[str, Dict], ground_truth: str = None) -> Dict:
        """
        Provide qualitative evaluation framework.
        
        Args:
            results: Dictionary of model responses
            ground_truth: Optional ground truth answer
            
        Returns:
            Qualitative evaluation template
        """
        evaluation = {}
        
        for model_name, data in results.items():
            response = data.get("response", "")
            evaluation[model_name] = {
                "response": response,
                "criteria": {
                    "relevance": "To be evaluated manually (1-5): Does the answer address the user's question?",
                    "accuracy": "To be evaluated manually (1-5): Is the information factually correct based on KG context?",
                    "naturalness": "To be evaluated manually (1-5): Does the response read naturally and fluently?",
                    "completeness": "To be evaluated manually (1-5): Is all relevant information from context included?",
                    "groundedness": "To be evaluated manually (1-5): Does it stay faithful to the KG context without hallucination?"
                },
                "notes": ""
            }
        
        return evaluation
    
    def compare_responses(self, results: Dict[str, Dict]) -> Dict:
        """
        Compare responses across models for analysis.
        
        Args:
            results: Dictionary of model responses
            
        Returns:
            Comparison analysis
        """
        comparison = {
            "total_models": len(results),
            "response_similarities": {},
            "length_variance": [],
            "common_elements": []
        }
        
        responses = {name: data.get("response", "") for name, data in results.items()}
        lengths = {name: len(resp) for name, resp in responses.items()}
        
        comparison["length_variance"] = lengths
        comparison["avg_length"] = sum(lengths.values()) / len(lengths) if lengths else 0
        
        # Find common keywords/phrases across responses
        if len(responses) >= 2:
            words_sets = []
            for resp in responses.values():
                words = set(resp.lower().split())
                words_sets.append(words)
            
            if words_sets:
                common_words = set.intersection(*words_sets)
                comparison["common_elements"] = list(common_words)[:20]  # Top 20 common words
        
        return comparison
    
    def save_evaluation(self, results: Dict, output_file: str):
        """Save evaluation results to a JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

