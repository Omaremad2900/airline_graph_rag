# Quick Start Guide

## Prerequisites

1. **Neo4j Database**: Ensure Neo4j is running with your airline knowledge graph from Milestone 2
2. **Python 3.8+**: Required for the project
3. **API Keys** (Optional): For LLM models (OpenAI, Anthropic, Google, OpenRouter, or HuggingFace)

## Setup Steps

### 1. Install Dependencies

```bash
cd airline_graph_rag
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your credentials
# At minimum, set:
# - NEO4J_URI
# - NEO4J_USERNAME
# - NEO4J_PASSWORD
```

### 3. (Optional) Initialize Embeddings

For embedding-based retrieval, you need to create embeddings first:

```bash
python scripts/initialize_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2
```

This will:
- Create feature vector embeddings for Journey nodes
- Store them in Neo4j
- Take a few minutes depending on data size

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Using the Application

### Basic Usage

1. **Connect to Neo4j**: Click "🔌 Connect to Neo4j" in the sidebar
2. **Select Retrieval Method**: Choose Baseline Only, Embeddings Only, or Both
3. **Select LLM Model**: Choose from available models (requires API keys)
4. **Enter Query**: Type your question in the input box
5. **Click Query**: View results, context, and LLM answer

### Example Queries

Try these queries to test the system:

- `Which flights have the worst delays?`
- `Show me flights from JFK to LAX`
- `What are the routes with low passenger satisfaction?`
- `Find flights departing from ORD`
- `Which routes have the most delays?`
- `Show me journeys with food satisfaction below 3`
- `What are the most popular routes?`
- `Tell me about flight AA123`
- `Compare performance of different passenger classes`
- `Recommend the best routes for on-time performance`

### Advanced Features

#### Model Comparison

1. Check "Compare Multiple Models" in sidebar
2. Select multiple models to compare
3. View side-by-side responses and metrics

#### Graph Visualization

1. After querying, check "Show Graph Visualization"
2. View the retrieved subgraph with nodes and relationships

#### View Cypher Queries

1. Expand "Cypher Queries Executed" section
2. See the actual queries that were run

## Troubleshooting

### Connection Issues

**Problem**: "Failed to connect to Neo4j"
- **Solution**: Check NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in .env
- Verify Neo4j is running: `neo4j status`

### No Embeddings Found

**Problem**: "No embeddings found" when using embedding retrieval
- **Solution**: Run `python scripts/initialize_embeddings.py` first

### LLM Not Available

**Problem**: "No LLM models available"
- **Solution**: Add at least one API key to .env file
- For free models, use OpenRouter or HuggingFace

### Import Errors

**Problem**: ModuleNotFoundError
- **Solution**: Ensure you're in the project directory and dependencies are installed
- Try: `pip install -r requirements.txt --upgrade`

## Testing

Run test queries from the test file:

```python
import json
with open('tests/test_queries.json') as f:
    test_queries = json.load(f)
    for test in test_queries:
        print(test['query'])
```

## Next Steps

1. **Customize Queries**: Add your own Cypher query templates in `retrieval/baseline.py`
2. **Add Entities**: Extend entity extraction in `preprocessing/entity_extractor.py`
3. **Configure Models**: Add more LLM models in `config.py`
4. **Evaluate Performance**: Use the evaluation framework in `utils/evaluation.py`

## Support

For issues or questions:
1. Check the README.md for detailed documentation
2. Review ARCHITECTURE.md for system design
3. Check the code comments for implementation details

