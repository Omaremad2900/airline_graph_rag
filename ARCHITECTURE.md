# System Architecture

## Overview

The Graph-RAG Airline Assistant is an end-to-end system that combines Neo4j Knowledge Graph with Large Language Models to provide airline company insights. The system follows a pipeline architecture with four main components.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (Streamlit)                │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              1. Input Preprocessing Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Intent     │  │   Entity     │  │  Embedding   │      │
│  │ Classification│  │ Extraction  │  │  Generation  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              2. Graph Retrieval Layer                        │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │   Baseline       │         │   Embeddings     │         │
│  │   (Cypher)       │         │   (Semantic)     │         │
│  │                  │         │                  │         │
│  │ • 10+ Query      │         │ • Feature Vector │         │
│  │   Templates      │         │   Embeddings     │         │
│  │ • Intent-based   │         │ • 2 Models       │         │
│  │   Routing        │         │ • Similarity      │         │
│  └──────────────────┘         └──────────────────┘         │
│           │                           │                     │
│           └───────────┬───────────────┘                     │
│                       ▼                                     │
│              Combine & Deduplicate                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              3. LLM Layer                                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │           Structured Prompt Builder                 │    │
│  │  • Context: Retrieved KG data                      │    │
│  │  • Persona: Airline insights assistant             │    │
│  │  • Task: Answer using context only                 │    │
│  └────────────────────────────────────────────────────┘    │
│                            │                                │
│                            ▼                                │
│  ┌────────────────────────────────────────────────────┐    │
│  │           Multi-Model LLM Support                  │    │
│  │  • OpenAI (GPT-3.5, GPT-4)                        │    │
│  │  • Anthropic (Claude)                              │    │
│  │  • Google (Gemini)                                 │    │
│  │  • OpenRouter (Mistral, Llama)                     │    │
│  │  • HuggingFace                                     │    │
│  └────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              4. Response & Visualization                     │
│  • LLM Answer Display                                       │
│  • Retrieved Context Table                                  │
│  • Graph Visualization                                       │
│  • Cypher Query Display                                     │
│  • Model Comparison Metrics                                 │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Input Preprocessing

#### Intent Classifier
- **Method**: Rule-based pattern matching with optional LLM enhancement
- **Intents**: 
  - `flight_search`: Finding flights by route, departure, arrival
  - `delay_analysis`: Analyzing delays and on-time performance
  - `passenger_satisfaction`: Querying satisfaction scores and feedback
  - `route_analysis`: Analyzing routes and connections
  - `journey_insights`: Journey details and statistics
  - `performance_metrics`: Overall performance statistics
  - `recommendation`: Getting recommendations
  - `general_question`: General queries

#### Entity Extractor
- **Method**: Named Entity Recognition (NER) using regex patterns
- **Entity Types**:
  - `AIRPORT`: Airport codes (3-letter IATA) and names
  - `FLIGHT`: Flight numbers (e.g., AA123)
  - `DATE`: Various date formats
  - `NUMBER`: Numeric values for thresholds

#### Embedding Generator
- **Models**: 
  - `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
  - `sentence-transformers/all-mpnet-base-v2` (768 dimensions)
- **Purpose**: Convert text to vectors for semantic similarity search

### 2. Graph Retrieval Layer

#### Baseline Retrieval (Cypher Queries)

**Query Templates by Intent:**

1. **Flight Search** (3 templates)
   - By route (departure + arrival)
   - By departure only
   - By arrival only

2. **Delay Analysis** (3 templates)
   - Flights with delays above threshold
   - Delays by route
   - Worst delayed flights

3. **Passenger Satisfaction** (3 templates)
   - Low-rated journeys
   - Satisfaction by passenger class
   - Poor-performing flights

4. **Route Analysis** (3 templates)
   - Popular routes
   - Route performance metrics
   - Multi-leg journeys

5. **Journey Insights** (2 templates)
   - Journey details by feedback ID
   - Loyalty passenger journeys

6. **Performance Metrics** (2 templates)
   - Overall statistics
   - Flight performance comparison

7. **Recommendation** (1 template)
   - Best routes by satisfaction and on-time performance

8. **General Question** (1 template)
   - Flight information by flight number

**Total: 18 query templates** (exceeds requirement of 10+)

#### Embedding-Based Retrieval

**Feature Vector Embeddings:**
- Combines multiple Journey properties into a single text representation:
  - Route (departure → arrival)
  - Flight details (number, fleet type)
  - Passenger class and loyalty level
  - Food satisfaction score
  - Arrival delay
  - Actual miles flown
  - Number of legs

- Embeddings stored in `Journey.feature_embedding` property
- Similarity search using cosine similarity
- Supports two embedding models for comparison

### 3. LLM Layer

#### Structured Prompts

**Format:**
```
[Persona Definition]

[Task Instructions]

Context from Knowledge Graph:
[Retrieved data in JSON format]

User Query: [Original query]

Answer:
```

**Persona**: "You are an Airline Company Flight Insights Assistant..."

**Task**: "Based on the context provided from the knowledge graph, answer the user's question accurately..."

#### Multi-Model Support

**Supported Providers:**
- OpenAI: GPT-3.5-turbo, GPT-4
- Anthropic: Claude 3 Haiku
- Google: Gemini Pro
- OpenRouter: Mistral-7B, Llama-3-8B
- HuggingFace: Various open-source models

**Metrics Tracked:**
- Response time
- Token usage (estimated)
- Response length

### 4. User Interface

**Features:**
- Query input with real-time processing
- Preprocessing results display (intent, entities)
- Retrieved context table
- LLM answer display
- Cypher query visualization
- Graph visualization (NetworkX + Pyvis)
- Model comparison (side-by-side)
- Retrieval method selection
- Query history

## Data Flow

1. **User Input** → Intent Classification + Entity Extraction
2. **Intent + Entities** → Route to appropriate Cypher query templates
3. **Query Execution** → Retrieve nodes, relationships, properties from Neo4j
4. **User Query** → Generate embedding → Semantic similarity search
5. **Combine Results** → Merge baseline + embedding results, deduplicate
6. **Build Prompt** → Structure context + persona + task
7. **LLM Generation** → Generate answer using retrieved context
8. **Display** → Show answer, context, queries, visualizations

## Experiments

### Experiment 1: Baseline Only
- Uses only Cypher queries
- Deterministic, exact matches
- Fast execution
- Limited to structured queries

### Experiment 2: Baseline + Embeddings (Hybrid)
- Combines Cypher queries with semantic search
- Handles both structured and semantic queries
- More comprehensive results
- Slightly slower due to similarity computation

## Evaluation Framework

### Quantitative Metrics
- Response time (seconds)
- Token usage (estimated)
- Response length (characters)
- Retrieval count (number of records)

### Qualitative Evaluation
- Relevance: Does the answer address the question?
- Accuracy: Is the information correct?
- Naturalness: Does it read naturally?
- Completeness: Is all relevant information included?

## Knowledge Graph Schema

**Nodes:**
- `Passenger`: record_locator, loyalty_program_level, generation
- `Journey`: feedback_ID, food_satisfaction_score, arrival_delay_minutes, actual_flown_miles, number_of_legs, passenger_class, feature_embedding
- `Flight`: flight_number, fleet_type_description
- `Airport`: station_code

**Relationships:**
- `(Passenger)-[:TAKES]->(Journey)`
- `(Journey)-[:ON]->(Flight)`
- `(Flight)-[:DEPARTS_FROM]->(Airport)`
- `(Flight)-[:ARRIVES_AT]->(Airport)`

## Performance Considerations

- **Embedding Initialization**: One-time process, can be run offline
- **Vector Index**: Optional Neo4j vector index for faster similarity search
- **Result Limiting**: Results capped at 50 records to manage context size
- **Context Size**: LLM context limited to top 30 records to avoid token limits
- **Caching**: Session state caching for components to avoid re-initialization

## Extensibility

The system is designed to be easily extensible:

- **New Intents**: Add patterns to `IntentClassifier`
- **New Entities**: Add extraction logic to `EntityExtractor`
- **New Queries**: Add templates to `BaselineRetriever.query_templates`
- **New Models**: Add configuration to `config.LLM_MODELS`
- **New Embeddings**: Add to `config.EMBEDDING_MODELS`

