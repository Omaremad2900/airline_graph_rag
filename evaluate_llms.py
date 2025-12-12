"""
Comprehensive LLM Model Evaluation Script for Milestone 3

This script evaluates and compares multiple LLM models on test queries,
providing both quantitative and qualitative metrics as required by the milestone.
"""

import json
import time
from typing import Dict, List
import pandas as pd
from datetime import datetime

from utils.neo4j_connector import Neo4jConnector
from preprocessing.intent_classifier import IntentClassifier
from preprocessing.entity_extractor import EntityExtractor
from retrieval.baseline import BaselineRetriever
from retrieval.embeddings import EmbeddingRetriever
from preprocessing.embedding import EmbeddingGenerator
from llm_layer.models import LLMManager
from llm_layer.prompts import build_prompt, get_persona, get_task_instruction
from utils.evaluation import ModelEvaluator


class LLMEvaluationRunner:
    """Runs comprehensive LLM evaluation experiments."""
    
    def __init__(self, neo4j_connector=None):
        """Initialize evaluation runner with components."""
        self.connector = neo4j_connector or Neo4jConnector()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.baseline_retriever = BaselineRetriever(self.connector)
        
        # Initialize embedding retrievers
        self.embedding_retrievers = {}
        for model_name in ["sentence-transformers/all-MiniLM-L6-v2"]:
            try:
                embedding_model = EmbeddingGenerator(model_name)
                self.embedding_retrievers[model_name] = EmbeddingRetriever(
                    self.connector, embedding_model
                )
            except Exception as e:
                print(f"Warning: Could not initialize embedding retriever for {model_name}: {e}")
        
        # Initialize LLM manager
        self.llm_manager = LLMManager()
        self.evaluator = ModelEvaluator(self.llm_manager)
        
    def retrieve_context(self, query: str, use_baseline: bool = True, 
                        use_embeddings: bool = True) -> tuple:
        """
        Retrieve context from KG using baseline and/or embeddings.
        
        Returns:
            Tuple of (combined_results, baseline_results, embedding_results)
        """
        # Preprocessing
        intent = self.intent_classifier.classify(query)
        entities = self.entity_extractor.extract_entities(query)
        
        baseline_results = []
        embedding_results = []
        
        # Baseline retrieval
        if use_baseline:
            try:
                baseline_results, _ = self.baseline_retriever.retrieve(intent, entities)
            except Exception as e:
                print(f"Baseline retrieval error: {e}")
        
        # Embedding retrieval
        if use_embeddings and self.embedding_retrievers:
            try:
                retriever = list(self.embedding_retrievers.values())[0]
                embedding_results = retriever.retrieve_by_similarity(query, top_k=10)
            except Exception as e:
                print(f"Embedding retrieval error: {e}")
        
        # Combine and deduplicate
        all_results = baseline_results + embedding_results
        seen = set()
        unique_results = []
        for r in all_results:
            key = str(sorted(r.items()))
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        return unique_results[:30], baseline_results, embedding_results
    
    def evaluate_models_on_query(self, query: str, models: List[str],
                                use_baseline: bool = True,
                                use_embeddings: bool = True) -> Dict:
        """
        Evaluate multiple models on a single query.
        
        Args:
            query: User query
            models: List of model names to compare
            use_baseline: Whether to use baseline retrieval
            use_embeddings: Whether to use embedding retrieval
            
        Returns:
            Dictionary with results and metrics
        """
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        # Retrieve context
        print("Retrieving context from KG...")
        context_results, baseline_results, embedding_results = self.retrieve_context(
            query, use_baseline, use_embeddings
        )
        
        print(f"Retrieved: {len(baseline_results)} baseline, "
              f"{len(embedding_results)} embedding, "
              f"{len(context_results)} total unique")
        
        if not context_results:
            print("⚠️  No context retrieved from KG")
            return {
                "query": query,
                "context_retrieved": 0,
                "models": {},
                "error": "No context retrieved"
            }
        
        # Format context for LLM
        context = json.dumps(context_results, indent=2)
        persona = get_persona()
        task = get_task_instruction()
        prompt = build_prompt(context, persona, task, query)
        
        # Compare models
        print(f"\nComparing {len(models)} models...")
        model_results = {}
        
        for model_name in models:
            if model_name not in self.llm_manager.available_models:
                print(f"⚠️  Model {model_name} not available (missing API key?)")
                continue
            
            print(f"  - Testing {model_name}...")
            try:
                model = self.llm_manager.get_model(model_name)
                start_time = time.time()
                response = model.invoke(prompt)
                elapsed = time.time() - start_time
                
                metrics = model.get_metrics()
                
                model_results[model_name] = {
                    "response": response,
                    "metrics": metrics,
                    "elapsed_time": elapsed
                }
                
                print(f"    ✓ Response time: {elapsed:.2f}s, "
                      f"Est. tokens: {metrics.get('estimated_tokens', 'N/A')}")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                model_results[model_name] = {
                    "response": "",
                    "error": str(e),
                    "metrics": {}
                }
        
        return {
            "query": query,
            "context_retrieved": len(context_results),
            "baseline_count": len(baseline_results),
            "embedding_count": len(embedding_results),
            "models": model_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def run_evaluation_suite(self, test_queries: List[Dict], models: List[str],
                            output_file: str = "llm_evaluation_results.json"):
        """
        Run full evaluation suite on multiple queries.
        
        Args:
            test_queries: List of query dictionaries with 'query' and optional 'intent'
            models: List of model names to compare
            output_file: Path to save results
        """
        print("\n" + "="*80)
        print("LLM EVALUATION SUITE")
        print("="*80)
        print(f"Test Queries: {len(test_queries)}")
        print(f"Models: {', '.join(models)}")
        print(f"Available Models: {', '.join(self.llm_manager.available_models)}")
        print("="*80)
        
        all_results = []
        
        for i, query_data in enumerate(test_queries, 1):
            query = query_data.get("query", "")
            if not query:
                continue
            
            print(f"\n[{i}/{len(test_queries)}] Processing query...")
            
            result = self.evaluate_models_on_query(
                query=query,
                models=models,
                use_baseline=True,
                use_embeddings=True
            )
            
            all_results.append(result)
        
        # Save results
        output_data = {
            "evaluation_date": datetime.now().isoformat(),
            "total_queries": len(test_queries),
            "models_tested": models,
            "results": all_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✅ Evaluation complete! Results saved to: {output_file}")
        print(f"{'='*80}")
        
        # Generate summary
        self.generate_summary(all_results, models)
        
        return all_results
    
    def generate_summary(self, results: List[Dict], models: List[str]):
        """Generate and print summary statistics."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        # Aggregate metrics
        model_stats = {model: {
            "total_queries": 0,
            "successful": 0,
            "failed": 0,
            "avg_response_time": [],
            "avg_tokens": [],
            "avg_response_length": []
        } for model in models}
        
        for result in results:
            for model_name, model_result in result.get("models", {}).items():
                if model_name not in model_stats:
                    continue
                
                stats = model_stats[model_name]
                stats["total_queries"] += 1
                
                if "error" in model_result:
                    stats["failed"] += 1
                else:
                    stats["successful"] += 1
                    stats["avg_response_time"].append(model_result.get("elapsed_time", 0))
                    
                    metrics = model_result.get("metrics", {})
                    stats["avg_tokens"].append(metrics.get("estimated_tokens", 0))
                    stats["avg_response_length"].append(len(model_result.get("response", "")))
        
        # Print summary
        print("\n{:<30} {:>10} {:>10} {:>10} {:>12} {:>12}".format(
            "Model", "Success", "Failed", "Avg Time", "Avg Tokens", "Avg Length"
        ))
        print("-" * 95)
        
        for model_name, stats in model_stats.items():
            if stats["total_queries"] == 0:
                continue
            
            avg_time = sum(stats["avg_response_time"]) / len(stats["avg_response_time"]) if stats["avg_response_time"] else 0
            avg_tokens = sum(stats["avg_tokens"]) / len(stats["avg_tokens"]) if stats["avg_tokens"] else 0
            avg_length = sum(stats["avg_response_length"]) / len(stats["avg_response_length"]) if stats["avg_response_length"] else 0
            
            print("{:<30} {:>10} {:>10} {:>9.2f}s {:>12.0f} {:>12.0f}".format(
                model_name[:28],
                stats["successful"],
                stats["failed"],
                avg_time,
                avg_tokens,
                avg_length
            ))
        
        print("="*80)


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LLM models for Graph-RAG")
    parser.add_argument("--queries", type=str, default="tests/test_queries.json",
                       help="Path to test queries JSON file")
    parser.add_argument("--models", type=str, nargs="+",
                       default=["gpt-3.5-turbo", "gpt-4", "claude-3-haiku-20240307"],
                       help="List of models to compare")
    parser.add_argument("--output", type=str, default="llm_evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--custom-query", type=str,
                       help="Run evaluation on a single custom query")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    print("Initializing evaluation runner...")
    try:
        connector = Neo4jConnector()
        if not connector.test_connection():
            print("❌ Failed to connect to Neo4j")
            return
        print("✅ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
        return
    
    runner = LLMEvaluationRunner(connector)
    
    # Custom query mode
    if args.custom_query:
        result = runner.evaluate_models_on_query(
            args.custom_query,
            args.models
        )
        print("\nResults:")
        for model_name, model_result in result.get("models", {}).items():
            print(f"\n{model_name}:")
            print(f"  Response: {model_result.get('response', 'N/A')[:200]}...")
            print(f"  Metrics: {model_result.get('metrics', {})}")
        return
    
    # Load test queries
    try:
        with open(args.queries, 'r') as f:
            test_queries = json.load(f)
        print(f"✅ Loaded {len(test_queries)} test queries")
    except Exception as e:
        print(f"❌ Error loading test queries: {e}")
        return
    
    # Run evaluation
    runner.run_evaluation_suite(
        test_queries=test_queries,
        models=args.models,
        output_file=args.output
    )
    
    print("\n✅ Evaluation complete!")
    print(f"Review results in: {args.output}")
    print("\nFor qualitative evaluation, review the responses in the JSON file")
    print("and manually assess relevance, accuracy, naturalness, and completeness.")


if __name__ == "__main__":
    main()
