"""Streamlit UI for the Graph-RAG Airline Assistant."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
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


def create_graph_visualization(results: List[Dict]) -> Network:
    """Create a network graph visualization from query results."""
    G = nx.DiGraph()
    
    # Extract nodes and edges from results
    for record in results[:20]:  # Limit for performance
        # Add nodes based on record keys
        for key, value in record.items():
            if isinstance(value, (str, int, float)):
                node_id = f"{key}_{value}"
                G.add_node(node_id, label=f"{key}: {value}", type=key)
        
        # Add edges between related nodes
        keys = list(record.keys())
        for i in range(len(keys) - 1):
            if keys[i] in record and keys[i+1] in record:
                source = f"{keys[i]}_{record[keys[i]]}"
                target = f"{keys[i+1]}_{record[keys[i+1]]}"
                if source in G and target in G:
                    G.add_edge(source, target)
    
    # Create Pyvis network
    net = Network(height="500px", width="100%", directed=True)
    net.from_nx(G)
    return net


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
        
        st.divider()
        
        # LLM Model Selection
        st.subheader("🤖 LLM Model")
        if st.session_state.llm_manager:
            available_models = st.session_state.llm_manager.list_available_models()
            if available_models:
                selected_model = st.selectbox(
                    "Choose LLM:",
                    available_models
                )
            else:
                st.warning("No LLM models available. Check API keys.")
                selected_model = None
        else:
            st.warning("Initialize connection first")
            selected_model = None
        
        # Compare Models Option
        compare_models = st.checkbox("Compare Multiple Models", value=False)
        if compare_models and st.session_state.llm_manager:
            available_models = st.session_state.llm_manager.list_available_models()
            selected_models = st.multiselect(
                "Select models to compare:",
                available_models,
                default=available_models[:3] if len(available_models) >= 3 else available_models
            )
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
            # Step 1: Preprocessing (with error handling)
            try:
                intent = st.session_state.intent_classifier.classify(user_query)
            except Exception as e:
                st.error(f"❌ Intent classification failed: {e}")
                intent = "general_question"  # Safe fallback
                st.warning("⚠️ Using default intent: general_question")
            
            try:
                entities = st.session_state.entity_extractor.extract_entities(user_query)
            except Exception as e:
                st.error(f"❌ Entity extraction failed: {e}")
                entities = {}  # Safe fallback
                st.warning("⚠️ No entities extracted")
            
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
                    embedding_retriever = st.session_state.embedding_retrievers[embedding_model_name]
                    embedding_results = embedding_retriever.retrieve_by_similarity(user_query, top_k=10)
                    embedding_results_by_model[embedding_model_name] = embedding_results
                
                # If comparing models, get results from all models
                if retrieval_method == "Both (Hybrid)":
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
            # Remove duplicates
            seen = set()
            unique_results = []
            for r in all_results:
                key = str(sorted(r.items()))
                if key not in seen:
                    seen.add(key)
                    unique_results.append(r)
            
            # Format context for LLM
            context = json.dumps(unique_results[:30], indent=2)  # Limit context size
            
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
                
                # Graph visualization
                if st.checkbox("Show Graph Visualization"):
                    try:
                        net = create_graph_visualization(unique_results)
                        net.save_graph("graph.html")
                        st.components.v1.html(open("graph.html", "r").read(), height=600)
                    except Exception as e:
                        st.warning(f"Graph visualization error: {e}")
            else:
                st.warning("No results retrieved from knowledge graph.")
            
            # Display Cypher queries
            if cypher_queries:
                with st.expander("🔧 Cypher Queries Executed", expanded=False):
                    for i, query_info in enumerate(cypher_queries, 1):
                        st.write(f"**Query {i}: {query_info['template']}** (Intent: {query_info['intent']})")
                        st.write(f"**Parameters:** {json.dumps(query_info['parameters'], indent=2)}")
                        st.write(f"**Results:** {query_info['result_count']} records")
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

