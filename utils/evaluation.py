"""Evaluation utilities for comparing LLM models."""
import json
from typing import Dict, List
from llm_layer.models import LLMManager


class ModelEvaluator:
    """Evaluates and compares LLM models."""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
    
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
            metrics[model_name] = {
                "response_time": model_metrics.get("response_time", 0),
                "estimated_tokens": model_metrics.get("estimated_tokens", 0),
                "response_length": len(data.get("response", ""))
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
                    "relevance": "To be evaluated manually",
                    "accuracy": "To be evaluated manually",
                    "naturalness": "To be evaluated manually",
                    "completeness": "To be evaluated manually"
                }
            }
        
        return evaluation
    
    def save_evaluation(self, results: Dict, output_file: str):
        """Save evaluation results to a JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

