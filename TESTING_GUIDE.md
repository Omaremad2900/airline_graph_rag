# Testing Guide for Graph Retrieval Layer

This guide explains how to test the graph retrieval layer implementation, including baseline Cypher queries and embeddings-based retrieval.

## Prerequisites

1. **Neo4j Database**: Must be running with your airline knowledge graph loaded
2. **Environment Setup**: `.env` file configured with Neo4j credentials
3. **Dependencies**: All Python packages installed (`pip install -r requirements.txt`)

## Quick Start Testing

### 1. Test Query Templates Coverage

Verify that all intent categories have query templates:

```bash
python test_retrieval.py templates
```

**Expected Output:**

- ✅ All 8 intent categories should have query templates
- ✅ Total of 18+ query templates (exceeds requirement of 10+)

### 2. Test Baseline Retrieval (Cypher Queries)

Test the baseline retrieval with Cypher queries:

```bash
python test_retrieval.py baseline
```

**What it tests:**

- Intent classification
- Entity extraction
- Cypher query execution
- Result retrieval

**Expected Output:**

- Successful connection to Neo4j
- Queries executed for different intents
- Results returned from knowledge graph

### 3. Test Embeddings Retrieval

Test semantic similarity search using embeddings:

```bash
# First, initialize embeddings (if not done already)
python scripts/initialize_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2

# Then test embeddings retrieval
python test_retrieval.py embeddings
```

**What it tests:**

- Embedding model initialization
- Semantic similarity search
- Query-to-journey matching

**Expected Output:**

- Embeddings found in Neo4j
- Similar journeys retrieved based on query semantics
- Similarity scores displayed

### 4. Test Hybrid Retrieval

Test combining baseline and embeddings:

```bash
python test_retrieval.py hybrid
```

**What it tests:**

- Both retrieval methods working together
- Result combination and deduplication
- Coverage comparison

### 5. Run All Tests

Run comprehensive test suite:

```bash
python test_retrieval.py
```

This runs all tests in sequence:

1. Query templates coverage
2. Baseline retrieval
3. Embeddings retrieval
4. Hybrid retrieval

## Interactive Testing

For interactive testing with custom queries:

```bash
python test_retrieval.py interactive
```

**Commands:**

- `baseline` - Switch to baseline-only mode
- `embeddings` - Switch to embeddings-only mode
- `both` - Switch to hybrid mode
- `quit` - Exit

**Example Session:**

```
Enter query: Find flights from JFK to LAX
Intent: flight_search
Entities: {"AIRPORT": [{"value": "JFK", "type": "AIRPORT_CODE"}, ...]}
Baseline: 5 results
Embeddings: 3 results
```

## Testing with Streamlit UI

### 1. Start the Application

```bash
streamlit run app.py
```

### 2. Test Baseline Only (Experiment 1)

1. Click "🔌 Connect to Neo4j" in sidebar
2. Select "Baseline Only" as retrieval method
3. Enter test queries:
   - `Find flights from JFK to LAX`
   - `Which flights have delays over 30 minutes?`
   - `Show me journeys with low satisfaction`
4. Check "Cypher Queries Executed" to see which queries ran
5. Verify results match the query intent

### 3. Test Baseline + Embeddings (Experiment 2)

1. Ensure embeddings are initialized:

   ```bash
   python scripts/initialize_embeddings.py --all
   ```

2. In the UI:

   - Select "Both (Hybrid)" as retrieval method
   - Choose an embedding model
   - Enter semantic queries:
     - `Show me journeys with good food and minimal delays`
     - `Find flights with high satisfaction ratings`
     - `Which journeys have long delays?`

3. Compare results:
   - View baseline results count
   - View embedding results count
   - Check combined unique results
   - Compare similarity scores

### 4. Test Different Intent Categories

Test queries for each intent category:

| Intent                   | Example Query                                   |
| ------------------------ | ----------------------------------------------- |
| `flight_search`          | "Find flights from ORD to LAX"                  |
| `delay_analysis`         | "Which flights have the worst delays?"          |
| `passenger_satisfaction` | "Show me journeys with low satisfaction"        |
| `route_analysis`         | "What are the most popular routes?"             |
| `journey_insights`       | "Tell me about journey ABC123"                  |
| `performance_metrics`    | "Compare performance of different classes"      |
| `recommendation`         | "Recommend best routes for on-time performance" |
| `general_question`       | "What is flight AA123?"                         |

## Testing Embedding Models

### Initialize Embeddings for Both Models

```bash
# Initialize for all models
python scripts/initialize_embeddings.py --all
```

This creates embeddings using:

- `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- `sentence-transformers/all-mpnet-base-v2` (768 dimensions)

### Compare Models in UI

1. Select "Both (Hybrid)" retrieval method
2. Switch between embedding models in the dropdown
3. Run the same query with different models
4. Compare:
   - Number of results
   - Average similarity scores
   - Result quality

## Test Cases from test_queries.json

The system includes predefined test queries in `tests/test_queries.json`:

```bash
# Load and test with predefined queries
python -c "
import json
from test_retrieval import test_baseline_retrieval
test_baseline_retrieval()
"
```

## Verification Checklist

### Baseline Retrieval ✅

- [ ] All 8 intent categories have query templates
- [ ] At least 10 query templates total (currently 18)
- [ ] Queries execute successfully for each intent
- [ ] Results are returned from Neo4j
- [ ] Entity extraction works (airports, flights, numbers)
- [ ] Query parameters are correctly built from entities

### Embeddings Retrieval ✅

- [ ] Embeddings initialized successfully
- [ ] Feature text descriptions created correctly
- [ ] Embeddings stored in Neo4j
- [ ] Semantic similarity search works
- [ ] Query embeddings match node embeddings
- [ ] Top-k results are relevant

### Hybrid Retrieval ✅

- [ ] Both methods execute successfully
- [ ] Results are combined correctly
- [ ] Duplicates are removed
- [ ] More comprehensive results than baseline alone

### Integration ✅

- [ ] Streamlit UI connects to Neo4j
- [ ] Preprocessing (intent + entities) works
- [ ] Retrieval methods can be selected
- [ ] Results display correctly
- [ ] Cypher queries are shown
- [ ] Graph visualization works

## Troubleshooting

### "Failed to connect to Neo4j"

**Solution:**

1. Check Neo4j is running: `neo4j status`
2. Verify credentials in `.env`:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   ```
3. Test connection: `python -c "from utils.neo4j_connector import Neo4jConnector; c = Neo4jConnector(); print(c.test_connection())"`

### "No embeddings found"

**Solution:**

```bash
# Initialize embeddings first
python scripts/initialize_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2
```

### "No results returned"

**Possible causes:**

1. Knowledge graph is empty - load your data first
2. Query parameters don't match data - check entity extraction
3. Intent classification incorrect - verify intent patterns

### "Import errors"

**Solution:**

```bash
pip install -r requirements.txt --upgrade
```

## Performance Testing

### Measure Query Performance

```python
import time
from test_retrieval import test_baseline_retrieval

start = time.time()
test_baseline_retrieval()
print(f"Time: {time.time() - start:.2f}s")
```

### Compare Retrieval Methods

1. Run same query with baseline only
2. Run same query with embeddings only
3. Run same query with hybrid
4. Compare:
   - Response time
   - Number of results
   - Result quality/relevance

## Next Steps

After testing:

1. **Evaluate Results**: Check if retrieved results are relevant to queries
2. **Tune Queries**: Adjust Cypher query templates if needed
3. **Improve Embeddings**: Modify feature text construction if results are poor
4. **Add More Tests**: Create additional test cases for your specific use cases

## Support

For issues:

1. Check error messages in test output
2. Verify Neo4j connection and data
3. Review `RETRIEVAL_IMPLEMENTATION.md` for implementation details
4. Check `ARCHITECTURE.md` for system design
