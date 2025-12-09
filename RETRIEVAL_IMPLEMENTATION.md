# Graph Retrieval Layer Implementation

## Overview

This document describes the implementation of the Graph Retrieval Layer, which supports two experiments:

1. **Baseline Only**: Using Cypher queries for deterministic retrieval
2. **Baseline + Embeddings**: Combining Cypher queries with semantic similarity search using vector embeddings

## 2.a. Baseline Retrieval (Cypher Queries)

### Implementation

The baseline retrieval is implemented in `retrieval/baseline.py` using the `BaselineRetriever` class.

### Query Templates

The system includes **18 Cypher query templates** organized by intent, exceeding the requirement of 10+ queries:

#### 1. Flight Search (3 templates)

- `by_route`: Find flights between two airports
- `by_departure`: Find flights departing from an airport
- `by_arrival`: Find flights arriving at an airport

#### 2. Delay Analysis (3 templates)

- `flights_with_delays`: Find flights with delays above a threshold
- `delays_by_route`: Analyze delays by route
- `worst_delayed_flights`: Identify worst performing flights by delay

#### 3. Passenger Satisfaction (3 templates)

- `low_rated_journeys`: Find journeys with low satisfaction scores
- `satisfaction_by_class`: Analyze satisfaction by passenger class
- `poor_performing_flights`: Identify flights with poor satisfaction or delays

#### 4. Route Analysis (3 templates)

- `popular_routes`: Find most popular routes
- `route_performance`: Analyze route performance metrics
- `multi_leg_journeys`: Find journeys with multiple legs

#### 5. Journey Insights (2 templates)

- `journey_details`: Get detailed information about a specific journey
- `loyalty_passenger_journeys`: Analyze journeys by loyalty program level

#### 6. Performance Metrics (2 templates)

- `overall_statistics`: Get overall system statistics
- `flight_performance`: Compare flight performance metrics

#### 7. Recommendation (1 template)

- `best_routes`: Recommend best routes by satisfaction and on-time performance

#### 8. General Question (1 template)

- `flight_info`: Get information about a specific flight

### Entity-Based Parameterization

The `_build_parameters()` method extracts entities from user input and maps them to Cypher query parameters:

- **AIRPORT entities**: Mapped to `departure_code` and `arrival_code` parameters
- **FLIGHT entities**: Mapped to `flight_number` parameter
- **NUMBER entities**: Mapped to thresholds like `min_delay` or `min_score`
- **DATE entities**: Can be used for time-based filtering (future enhancement)

### Query Execution

The `retrieve()` method:

1. Selects appropriate query templates based on classified intent
2. Builds parameters from extracted entities
3. Executes matching queries
4. Returns results along with executed query information for transparency

## 2.b. Embeddings-Based Retrieval

### Implementation

The embeddings-based retrieval is implemented in `retrieval/embeddings.py` using the `EmbeddingRetriever` class.

### Approach: Feature Vector Embeddings

The system uses **Feature Vector Embeddings** approach. For the Airline theme without textual features, the system constructs text descriptions from numerical properties (e.g., "Journey: X, Class: Y, Food: Z, Delay: W"), then embeds these text descriptions using sentence-transformers models.

### Text Description Construction

The `_create_feature_text()` method creates text descriptions from Journey numerical properties:

**Components included:**

- Journey identifier (feedback_ID)
- Route information (departure → arrival airports)
- Flight details (flight number, fleet type)
- Passenger class (Economy, Business, First)
- Loyalty program level
- Food satisfaction score
- Arrival delay (with context: early, on-time, or delayed)
- Actual miles flown
- Number of legs (direct flight vs. multiple legs)

**Example text description:**

```
Journey ABC123. from ORD to LAX. Flight AA123. Fleet type Boeing 737. Class Economy. Loyalty Gold. Food satisfaction 4. Arrived on time. Miles 1745. Direct flight
```

### Multiple Embedding Models Support

The system supports **two embedding models** for comparison:

1. **sentence-transformers/all-MiniLM-L6-v2**

   - Dimension: 384
   - Fast, lightweight model

2. **sentence-transformers/all-mpnet-base-v2**
   - Dimension: 768
   - More accurate, larger model

### Model-Specific Storage

Each embedding model stores embeddings in a separate property to allow comparison:

- Model 1: `feature_embedding_sentence_transformers_all_MiniLM_L6_v2`
- Model 2: `feature_embedding_sentence_transformers_all_mpnet_base_v2`

This allows both models to coexist in the same graph and be compared directly.

### Embedding Process

1. **Text Construction**: For each Journey node, create a text description from its numerical properties
2. **Embedding Generation**: Use the selected embedding model to convert the text description into a vector
3. **Storage**: Store the embedding vector in the Journey node's model-specific property

### Similarity Search

The `retrieve_by_similarity()` method:

1. Embeds the user query text using the selected embedding model
2. Retrieves all Journey nodes with embeddings for that model
3. Computes cosine similarity between query embedding and node embeddings
4. Returns top-k most similar journeys

The semantic similarity search captures the meaning of the query and matches it with journeys that have similar characteristics, even if the exact keywords don't match.

## Initialization

### Embedding Initialization Script

The `scripts/initialize_embeddings.py` script supports:

1. **Single Model Initialization**:

   ```bash
   python scripts/initialize_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2
   ```

2. **All Models Initialization**:
   ```bash
   python scripts/initialize_embeddings.py --all
   ```

The script:

- Creates feature embeddings for all Journey nodes
- Stores them in model-specific properties
- Creates vector indexes for fast similarity search (optional)

## Integration in Application

### Experiment 1: Baseline Only

When "Baseline Only" is selected:

- Only Cypher queries are executed
- Results are deterministic and based on exact matches
- Fast execution
- Limited to structured queries

### Experiment 2: Baseline + Embeddings (Hybrid)

When "Both (Hybrid)" is selected:

- Both Cypher queries and semantic search are executed
- Results are combined and deduplicated
- Supports both structured and semantic queries
- Can compare results from different embedding models

### Query Tracking

The system tracks and displays:

- Which Cypher queries were executed
- Parameters used for each query
- Number of results from each query
- Comparison metrics between embedding models

## Usage Examples

### Example 1: Flight Search (Baseline)

**Query**: "Find flights from ORD to LAX"

1. Intent classified as: `flight_search`
2. Entities extracted: `AIRPORT: [ORD, LAX]`
3. Cypher query executed: `by_route` template
4. Results: List of flights matching the route

### Example 2: Semantic Search (Embeddings)

**Query**: "Show me journeys with good food and minimal delays"

1. Query embedded using selected model
2. Similarity search finds journeys with:
   - High food satisfaction scores
   - Low arrival delays
3. Results ranked by similarity score

### Example 3: Hybrid Retrieval

**Query**: "Flights from Chicago with delays"

1. Baseline: Executes `by_departure` + `flights_with_delays` queries
2. Embeddings: Semantic search for "delays" and "Chicago"
3. Results combined and deduplicated
4. Both retrieval methods contribute to final answer

## Performance Considerations

- **Baseline queries**: Fast, deterministic, exact matches
- **Embedding search**: Slower due to similarity computation, but handles semantic queries
- **Result limiting**: Results capped at 50 for baseline, top-k for embeddings
- **Context size**: Limited to top 30 records for LLM context

## Future Enhancements

1. **Node Embeddings**: Alternative approach using graph embedding techniques (Node2Vec, GraphSAGE)
2. **Vector Index Optimization**: Use Neo4j's native vector index for faster similarity search
3. **Hybrid Scoring**: Combine Cypher query scores with similarity scores for better ranking
4. **Query Caching**: Cache frequently used queries and embeddings
5. **Entity Linking**: Better mapping of airport names to codes
