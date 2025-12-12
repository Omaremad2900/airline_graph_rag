# Graph-RAG Airline Travel Assistant

An end-to-end Graph-RAG system for airline company insights using Neo4j Knowledge Graph and LLMs.

## Features

- **Intent Classification**: Routes queries to appropriate retrieval strategies
- **Entity Extraction**: Identifies flights, airports, passengers, journeys, and routes
- **Dual Retrieval**: Baseline Cypher queries + Embedding-based semantic search
- **Multi-LLM Support**: Compare GPT-3.5, GPT-4, Claude, Gemini, and open-source models
- **Interactive UI**: Streamlit interface with graph visualization and query transparency

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. Ensure Neo4j is running with your airline knowledge graph from Milestone 2

4. (Optional) Initialize embeddings for embedding-based retrieval:
```bash
python scripts/initialize_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

## LLM Model Evaluation

For Milestone 3 requirement 3.d (compare 3+ models with qualitative and quantitative evaluation):

```bash
# Run comprehensive LLM evaluation
python evaluate_llms.py

# Custom query evaluation
python evaluate_llms.py --custom-query "Which flights have the worst delays?"

# Specify models to compare
python evaluate_llms.py --models gpt-3.5-turbo claude-3-haiku-20240307 mistralai/mistral-7b-instruct
```

See `LLM_EVALUATION_GUIDE.md` for detailed instructions on running evaluations and interpreting results.

## Project Structure

```
.
├── app.py                 # Streamlit UI
├── config.py             # Configuration settings
├── preprocessing/        # Input preprocessing modules
│   ├── __init__.py
│   ├── intent_classifier.py
│   ├── entity_extractor.py
│   └── embedding.py
├── retrieval/            # Graph retrieval modules
│   ├── __init__.py
│   ├── baseline.py      # Cypher query templates (10+ queries)
│   └── embeddings.py    # Embedding-based retrieval
├── llm_layer/           # LLM integration
│   ├── __init__.py
│   ├── models.py        # LLM model wrappers
│   └── prompts.py       # Structured prompts
├── utils/               # Utility functions
│   ├── __init__.py
│   ├── neo4j_connector.py
│   └── evaluation.py
├── scripts/             # Utility scripts
│   └── initialize_embeddings.py
└── tests/               # Test cases and evaluation
    └── test_queries.json
```

## System Architecture

1. **Input Preprocessing**: 
   - Intent Classification (rule-based with optional LLM enhancement)
   - Entity Extraction (NER for airports, flights, dates, numbers)
   - Input Embedding (for semantic similarity search)

2. **Graph Retrieval Layer**:
   - **Baseline**: 10+ Cypher query templates covering:
     - Flight search (by route, departure, arrival)
     - Delay analysis (worst delays, delays by route)
     - Passenger satisfaction (low ratings, class analysis)
     - Route analysis (popular routes, performance)
     - Journey insights (details, loyalty passengers)
     - Performance metrics (overall stats, flight performance)
     - Recommendations (best routes)
   - **Embeddings**: Semantic similarity search using:
     - Feature vector embeddings (combining journey properties)
     - Two embedding models for comparison (MiniLM-L6-v2, MPNet-Base-v2)

3. **LLM Layer**:
   - Structured prompts (Context + Persona + Task)
   - Multi-model support (OpenAI, Anthropic, Google, OpenRouter, HuggingFace)
   - Quantitative and qualitative evaluation

4. **UI Features**:
   - Query input and processing
   - Retrieved context display
   - LLM answer display
   - Cypher query visualization
   - Graph visualization
   - Model comparison
   - Retrieval method selection (baseline, embeddings, hybrid)

## Example Queries

- "Which flights have the worst delays?"
- "Show me flights from JFK to LAX"
- "What are the routes with low passenger satisfaction?"
- "Find flights departing from ORD"
- "Which routes have the most delays?"
- "Show me journeys with food satisfaction below 3"
- "What are the most popular routes?"
- "Tell me about flight AA123"
- "Compare performance of different passenger classes"
- "Recommend the best routes for on-time performance"

## Configuration

Edit `config.py` to:
- Add/remove LLM models
- Configure embedding models
- Adjust intent categories
- Modify entity types

## Experiments

The system supports two retrieval experiments:
1. **Baseline Only**: Uses only Cypher queries
2. **Baseline + Embeddings**: Hybrid approach combining both methods

Compare results in the UI by selecting different retrieval methods.

## Evaluation

The system includes:
- **Quantitative Metrics**: Response time, token usage, response length
- **Qualitative Evaluation**: Framework for manual assessment of relevance, accuracy, naturalness, completeness

Use the model comparison feature in the UI to evaluate multiple LLMs side-by-side.

## Requirements Met

✅ **Input Preprocessing**
- Intent Classification (rule-based + optional LLM)
- Entity Extraction (NER for airline entities)
- Input Embedding (for semantic search)

✅ **Graph Retrieval Layer**
- Baseline: 10+ Cypher query templates
- Embeddings: 2 different embedding models
- Feature vector embeddings for Journey nodes

✅ **LLM Layer**
- Structured prompts (Context + Persona + Task)
- Multi-model support (6 models across 5 providers)
- Quantitative and qualitative evaluation
- Comprehensive evaluation script (`evaluate_llms.py`)
- Cost estimation and performance metrics

✅ **UI (Streamlit)**
- View KG-retrieved context
- View final LLM answer
- Cypher queries executed
- Graph visualization
- Model selection dropdown
- Retrieval method selection

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide for setup and basic usage
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design
- **[PREPROCESSING.md](PREPROCESSING.md)** - Input preprocessing details
- **[RETRIEVAL_IMPLEMENTATION.md](RETRIEVAL_IMPLEMENTATION.md)** - Graph retrieval layer
- **[LLM_LAYER.md](LLM_LAYER.md)** - **LLM layer implementation (Milestone 3)**
- **[LLM_EVALUATION_GUIDE.md](LLM_EVALUATION_GUIDE.md)** - **Step-by-step evaluation guide**
- **[LLM_QUICK_REFERENCE.md](LLM_QUICK_REFERENCE.md)** - **Quick reference for LLM layer**
- **[FAISS_MIGRATION.md](FAISS_MIGRATION.md)** - FAISS vector storage migration
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Testing procedures

## Milestone 3 - LLM Layer Deliverables

### Requirement 3: LLM Layer Implementation

✅ **3.a Context Combination**: 
- Merges baseline + embedding results
- Deduplicates records
- Formats for LLM consumption
- Code: `app.py` lines 216-232

✅ **3.b Structured Prompt**: 
- Context + Persona + Task structure
- Grounds LLM in KG data
- Code: `llm_layer/prompts.py`

✅ **3.c Multi-Model Support**: 
- 6 models across 5 providers
- OpenAI, Anthropic, Google, OpenRouter, HuggingFace
- Code: `llm_layer/models.py`

✅ **3.d Evaluation Framework**:
- **Quantitative**: Response time, tokens, cost, speed
- **Qualitative**: Relevance, accuracy, naturalness, completeness, groundedness
- Evaluation script: `evaluate_llms.py`
- Sample results: `llm_evaluation_results_SAMPLE.json`

**Run evaluation**: 
```bash
python evaluate_llms.py --models gpt-3.5-turbo claude-3-haiku-20240307 mistralai/mistral-7b-instruct
```

**See**: [LLM_EVALUATION_GUIDE.md](LLM_EVALUATION_GUIDE.md) for detailed instructions.

