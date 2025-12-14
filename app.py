"""Streamlit UI for the Graph-RAG Airline Assistant."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from typing import Dict, List

from config import LLM_MODELS, EMBEDDING_MODELS
from utils.neo4j_connector import Neo4jConnector
from preprocessing.intent_classifier import IntentClassifier
from preprocessing.entity_extractor import EntityExtractor
from preprocessing.embedding import EmbeddingGenerator
from retrieval.baseline import BaselineRetriever
from retrieval.embeddings import EmbeddingRetriever
from llm_layer.models import LLMManager
from llm_layer.prompts import build_prompt, get_persona, get_task_instruction
from utils.evaluation import ModelEvaluator


# Page configuration
st.set_page_config(
    page_title="Graph-RAG Airline Assistant",
    page_icon="✈️",
    layout="wide"
)

# Initialize session state
if 'connector' not in st.session_state:
    st.session_state.connector = None
if 'baseline_retriever' not in st.session_state:
    st.session_state.baseline_retriever = None
if 'embedding_retrievers' not in st.session_state:
    st.session_state.embedding_retrievers = {}
if 'llm_manager' not in st.session_state:
    st.session_state.llm_manager = None
if 'intent_classifier' not in st.session_state:
    st.session_state.intent_classifier = IntentClassifier()
if 'entity_extractor' not in st.session_state:
    st.session_state.entity_extractor = EntityExtractor()
if 'query_history' not in st.session_state:
    st.session_state.query_history = []


def initialize_components():
    """Initialize Neo4j connection and components."""
    try:
        if st.session_state.connector is None:
            st.session_state.connector = Neo4jConnector()
            if st.session_state.connector.test_connection():
                st.success("✅ Connected to Neo4j")
                st.session_state.baseline_retriever = BaselineRetriever(st.session_state.connector)
                
                # Initialize embedding retrievers for different models
                for model_name in EMBEDDING_MODELS.keys():
                    embedding_model = EmbeddingGenerator(model_name)
                    st.session_state.embedding_retrievers[model_name] = EmbeddingRetriever(
                        st.session_state.connector,
                        embedding_model
                    )
                
                # Initialize LLM manager
                st.session_state.llm_manager = LLMManager()
                return True
            else:
                st.error("❌ Failed to connect to Neo4j. Please check your connection settings.")
                return False
        return True
    except Exception as e:
        st.error(f"❌ Error initializing components: {e}")
        return False


def format_cypher_query(query: str) -> str:
    """Format Cypher query for display."""
    return query.strip()


def main():
    """Main Streamlit application."""
    st.title("✈️ Graph-RAG Airline Travel Assistant")
    st.markdown("**Airline Company Flight Insights Assistant** - Powered by Neo4j Knowledge Graph & LLMs")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Neo4j Connection Status
        if st.button("🔌 Connect to Neo4j"):
            with st.spinner("Connecting to Neo4j..."):
                initialize_components()
        
        if st.session_state.connector:
            st.success("Connected")
        else:
            st.warning("Not connected")
        
        st.divider()
        
        # Retrieval Method Selection
        st.subheader("🔍 Retrieval Method")
        retrieval_method = st.radio(
            "Choose retrieval method:",
            ["Baseline Only", "Embeddings Only", "Both (Hybrid)"],
            index=2
        )
        
        # Embedding Model Selection (if embeddings enabled)
        embedding_model_name = None
        if retrieval_method in ["Embeddings Only", "Both (Hybrid)"]:
            embedding_model_name = st.selectbox(
                "Embedding Model:",
                list(EMBEDDING_MODELS.keys()),
                format_func=lambda x: EMBEDDING_MODELS[x]["name"]
            )
            
            # Checkbox for comparing embedding models (separate from hybrid)
            compare_embedding_models = st.checkbox("Compare Embedding Models", value=False)
        else:
            compare_embedding_models = False
        
        st.divider()
        
        # LLM Provider Selection (Google, Groq, OpenRouter)
        st.subheader("🤖 LLM Provider")
        if st.session_state.llm_manager:
            # Filter models by provider (google, groq, openrouter)
            available_models = st.session_state.llm_manager.list_available_models()
            
            # Group models by provider
            google_models = [m for m in available_models if LLM_MODELS.get(m, {}).get("provider") == "google"]
            groq_models = [m for m in available_models if LLM_MODELS.get(m, {}).get("provider") == "groq"]
            openrouter_models = [m for m in available_models if LLM_MODELS.get(m, {}).get("provider") == "openrouter"]
            
            # Create provider options
            provider_options = []
            if google_models:
                provider_options.append("Google")
            if groq_models:
                provider_options.append("Groq")
            if openrouter_models:
                provider_options.append("OpenRouter")
            
            if provider_options:
                selected_provider = st.radio(
                    "Choose LLM Provider:",
                    provider_options,
                    index=0 if provider_options else None
                )
                
                # Get models for selected provider
                if selected_provider == "Google":
                    provider_models = google_models
                elif selected_provider == "Groq":
                    provider_models = groq_models
                elif selected_provider == "OpenRouter":
                    provider_models = openrouter_models
                else:
                    provider_models = []
                
                # Select specific model from provider (if multiple available)
                if len(provider_models) > 1:
                    selected_model = st.selectbox(
                        f"Choose {selected_provider} Model:",
                        provider_models,
                        format_func=lambda x: LLM_MODELS.get(x, {}).get("name", x) if "name" in LLM_MODELS.get(x, {}) else x
                    )
                elif len(provider_models) == 1:
                    selected_model = provider_models[0]
                    st.info(f"Using: {provider_models[0]}")
                else:
                    selected_model = None
                    st.warning(f"No {selected_provider} models available. Check API keys.")
            else:
                st.warning("No LLM providers available. Check API keys for Google, Groq, or OpenRouter.")
                selected_model = None
        else:
            st.warning("Initialize connection first")
            selected_model = None
        
        # Compare Models Option (across the three providers)
        compare_models = st.checkbox("Compare All Three Providers", value=False)
        if compare_models and st.session_state.llm_manager:
            available_models = st.session_state.llm_manager.list_available_models()
            # Group models by provider for comparison
            comp_google_models = [m for m in available_models if LLM_MODELS.get(m, {}).get("provider") == "google"]
            comp_groq_models = [m for m in available_models if LLM_MODELS.get(m, {}).get("provider") == "groq"]
            comp_openrouter_models = [m for m in available_models if LLM_MODELS.get(m, {}).get("provider") == "openrouter"]
            # Get one model from each provider
            comparison_models = []
            if comp_google_models:
                comparison_models.append(comp_google_models[0])
            if comp_groq_models:
                comparison_models.append(comp_groq_models[0])
            if comp_openrouter_models:
                comparison_models.append(comp_openrouter_models[0])
            selected_models = comparison_models
        else:
            selected_models = []
    
    # Main content area
    if not st.session_state.connector:
        st.info("👈 Please connect to Neo4j in the sidebar to begin.")
        return
    
    # Query input
    st.header("💬 Ask a Question")
    user_query = st.text_input(
        "Enter your question about flights, delays, passenger satisfaction, routes, or journey insights:",
        placeholder="e.g., Which flights have the worst delays? or Show me routes with low passenger satisfaction"
    )
    
    if st.button("🔍 Query", type="primary") and user_query:
        with st.spinner("Processing query..."):
            # Step 1: Preprocessing
            intent = st.session_state.intent_classifier.classify(user_query)
            entities = st.session_state.entity_extractor.extract_entities(user_query)
            
            # Display preprocessing results
            with st.expander("🔍 Preprocessing Results", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Intent:**", intent)
                with col2:
                    st.write("**Entities:**", json.dumps(entities, indent=2))
            
            # Step 2: Retrieval
            baseline_results = []
            embedding_results = []
            cypher_queries = []
            embedding_results_by_model = {}
            
            if retrieval_method in ["Baseline Only", "Both (Hybrid)"]:
                baseline_results, executed_queries = st.session_state.baseline_retriever.retrieve(intent, entities)
                cypher_queries = executed_queries
            
            if retrieval_method in ["Embeddings Only", "Both (Hybrid)"]:
                if embedding_model_name:
                    try:
                        embedding_retriever = st.session_state.embedding_retrievers[embedding_model_name]
                        embedding_results = embedding_retriever.retrieve_by_similarity(user_query, top_k=10)
                        embedding_results_by_model[embedding_model_name] = embedding_results
                    except Exception as e:
                        st.warning(f"Embedding retrieval failed: {e}")
                        embedding_results = []
                
                # If comparing embedding models, get results from all models
                if compare_embedding_models:
                    for model_name in EMBEDDING_MODELS.keys():
                        if model_name != embedding_model_name:
                            try:
                                retriever = st.session_state.embedding_retrievers[model_name]
                                results = retriever.retrieve_by_similarity(user_query, top_k=10)
                                embedding_results_by_model[model_name] = results
                            except Exception as e:
                                st.warning(f"Could not retrieve with model {model_name}: {e}")
            
            # Combine results
            all_results = baseline_results + embedding_results
            
            # Add source metadata to results
            for r in baseline_results:
                r["source"] = "baseline"
            for r in embedding_results:
                r["source"] = "embeddings"
            
            # Remove duplicates using stable keys
            seen = set()
            unique_results = []
            for r in all_results:
                # Use stable key: flight_number > feedback_id > full dict stringify
                if isinstance(r, dict):
                    if "flight_number" in r:
                        key = f"flight_{r['flight_number']}"
                    elif "feedback_id" in r or "feedback_ID" in r:
                        fid = r.get("feedback_id") or r.get("feedback_ID")
                        key = f"journey_{fid}"
                    else:
                        key = str(sorted(r.items()))
                else:
                    key = str(r)
                
                if key not in seen:
                    seen.add(key)
                    unique_results.append(r)
            
            # Score and sort hybrid results (embeddings first by similarity, then baseline)
            def hybrid_score(r):
                """Score for hybrid ranking: embeddings by similarity, baseline gets 0.5"""
                if "similarity_score" in r:
                    return float(r["similarity_score"])
                return 0.5
            
            unique_results.sort(key=hybrid_score, reverse=True)
            
            # Normalize record keys before building context
            def normalize_record(r):
                """Normalize record keys for consistent schema"""
                out = dict(r)
                
                # Normalize departure/arrival airport codes
                if "departure_airport" in out and "departure" not in out:
                    out["departure"] = out["departure_airport"]
                if "arrival_airport" in out and "arrival" not in out:
                    out["arrival"] = out["arrival_airport"]
                
                # Normalize flight number
                if "flight_flight_number" in out and "flight_number" not in out:
                    out["flight_number"] = out["flight_flight_number"]
                elif "flight_number" not in out:
                    # Try to extract from flight_ prefixed keys
                    for k, v in out.items():
                        if k.startswith("flight_") and "number" in k.lower():
                            out["flight_number"] = v
                            break
                
                # Normalize feedback_id
                if "feedback_ID" in out and "feedback_id" not in out:
                    out["feedback_id"] = out["feedback_ID"]
                
                # Ensure source is set
                if "source" not in out:
                    out["source"] = "unknown"
                
                return out
            
            # Normalize all records
            normalized_results = [normalize_record(r) for r in unique_results]
            
            # Format context for LLM - reduce size to avoid token limits
            # Limit to 10 records and truncate large fields to prevent context overflow
            max_records = 10
            max_field_length = 150  # Max characters per field
            truncated_results = []
            
            # Priority fields to keep
            priority_fields = ["flight_number", "departure", "arrival", "avg_delay", "max_delay", 
                             "journey_count", "avg_satisfaction", "similarity_score", "source",
                             "feedback_id", "food_satisfaction_score", "arrival_delay_minutes"]
            
            for record in normalized_results[:max_records]:
                truncated_record = {}
                # Add priority fields first
                for key in priority_fields:
                    if key in record:
                        value = record[key]
                        if isinstance(value, str) and len(value) > max_field_length:
                            truncated_record[key] = value[:max_field_length] + "..."
                        else:
                            truncated_record[key] = value
                
                # Add other fields (truncated)
                for key, value in record.items():
                    if key not in truncated_record:
                        if isinstance(value, str):
                            if len(value) > max_field_length:
                                truncated_record[key] = value[:max_field_length] + "..."
                            else:
                                truncated_record[key] = value
                        elif isinstance(value, (dict, list)):
                            str_value = json.dumps(value)
                            if len(str_value) > max_field_length:
                                if isinstance(value, list):
                                    truncated_record[key] = f"[List with {len(value)} items]"
                                else:
                                    truncated_record[key] = f"{{Object with {len(value)} keys}}"
                            else:
                                truncated_record[key] = value
                        else:
                            truncated_record[key] = value
                truncated_results.append(truncated_record)
            
            # Use compact JSON to save tokens
            context = json.dumps(truncated_results, separators=(',', ':'))  # No extra spaces
            
            # Step 3: LLM Generation
            if selected_model or compare_models:
                persona = get_persona()
                task = get_task_instruction()
                prompt = build_prompt(context, persona, task, user_query)
                
                if compare_models and selected_models:
                    # Compare multiple models
                    st.subheader("📊 Model Comparison")
                    comparison_results = st.session_state.llm_manager.compare_models(prompt, selected_models)
                    
                    # Display comparison
                    tabs = st.tabs([f"Model: {name}" for name in selected_models])
                    for idx, (model_name, result) in enumerate(comparison_results.items()):
                        with tabs[idx]:
                            st.write("**Response:**")
                            st.write(result["response"])
                            
                            st.write("**Metrics:**")
                            metrics_df = pd.DataFrame([result["metrics"]])
                            st.dataframe(metrics_df, use_container_width=True)
                    
                    # Quantitative comparison
                    evaluator = ModelEvaluator(st.session_state.llm_manager)
                    quant_metrics = evaluator.evaluate_quantitative(comparison_results)
                    
                    with st.expander("📈 Quantitative Metrics Comparison"):
                        metrics_comparison = pd.DataFrame(quant_metrics).T
                        st.dataframe(metrics_comparison)
                        
                        # Visualization
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=list(quant_metrics.keys()),
                            y=[m["response_time"] for m in quant_metrics.values()],
                            name="Response Time (s)"
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                
                elif selected_model:
                    # Single model
                    model = st.session_state.llm_manager.get_model(selected_model)
                    if model:
                        response = model.invoke(prompt)
                        metrics = model.get_metrics()
                        
                        st.subheader("💡 Answer")
                        st.write(response)
                        
                        with st.expander("📊 Model Metrics"):
                            st.json(metrics)
            
            # Display retrieved context
            st.subheader("📚 Retrieved Knowledge Graph Context")
            
            if unique_results:
                # Display as table
                df = pd.DataFrame(unique_results)
                st.dataframe(df, use_container_width=True)
                
                # Display statistics
                with st.expander("📊 Context Statistics"):
                    st.write(f"Total records retrieved: {len(unique_results)}")
                    st.write(f"Baseline results: {len(baseline_results)}")
                    st.write(f"Embedding results: {len(embedding_results)}")
            else:
                st.warning("No results retrieved from knowledge graph.")
            
            # Display Cypher queries
            if cypher_queries:
                with st.expander("🔧 Cypher Queries Executed", expanded=False):
                    for i, query_info in enumerate(cypher_queries, 1):
                        st.write(f"**Query {i}: {query_info['template']}** (Intent: {query_info['intent']})")
                        st.write(f"**Parameters:** {json.dumps(query_info['parameters'], indent=2)}")
                        st.write(f"**Results:** {query_info['result_count']} records")
                        # Show warnings if airport codes are invalid
                        if "warnings" in query_info and query_info["warnings"]:
                            for warning in query_info["warnings"]:
                                st.warning(f"⚠️ {warning}")
                        st.code(query_info['query'], language="cypher")
                        if i < len(cypher_queries):
                            st.divider()
            
            # Display embedding model comparison if multiple models used
            if len(embedding_results_by_model) > 1:
                with st.expander("🔍 Embedding Model Comparison", expanded=False):
                    comparison_data = []
                    for model_name, results in embedding_results_by_model.items():
                        model_display = EMBEDDING_MODELS[model_name]["name"]
                        comparison_data.append({
                            "Model": model_display,
                            "Results Count": len(results),
                            "Avg Similarity": sum(r.get("similarity_score", 0) for r in results) / len(results) if results else 0
                        })
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
            
            # Save to history
            st.session_state.query_history.append({
                "query": user_query,
                "intent": intent,
                "entities": entities,
                "results_count": len(unique_results)
            })
    
    # Query History
    if st.session_state.query_history:
        st.divider()
        st.subheader("📜 Query History")
        history_df = pd.DataFrame(st.session_state.query_history)
        st.dataframe(history_df, use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    **System Architecture:**
    - Input Preprocessing → Graph Retrieval → LLM Layer → Response
    
    **Features:**
    - Intent Classification & Entity Extraction
    - Baseline Cypher Queries + Embedding-based Semantic Search
    - Multi-LLM Support with Comparison
    - Graph Visualization & Query Transparency
    """)


if __name__ == "__main__":
    main()

