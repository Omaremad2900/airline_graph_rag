# Input Preprocessing Pipeline

This document explains the input preprocessing pipeline used in the Graph-RAG Airline Travel Assistant system.

## Overview

The preprocessing pipeline consists of three main components that work together to prepare user queries for retrieval and generation:

1. **Intent Classification** - Determines the user's intent from their query
2. **Entity Extraction** - Extracts structured entities (airports, flights, dates, etc.)
3. **Embedding Generation** - Creates vector embeddings for semantic similarity search

## Architecture

```
User Query
    ↓
┌─────────────────────────────────────┐
│  1. Intent Classifier               │ → Intent Category
│     (Pattern-based or LLM-based)    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. Entity Extractor                │ → Structured Entities
│     (Rule-based or LLM-based)       │   (Airports, Flights, Dates)
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. Embedding Generator             │ → Vector Embedding
│     (Sentence Transformers)          │   (384 or 768 dimensions)
└─────────────────────────────────────┘
    ↓
Processed Query (Ready for Retrieval)
```

## Component Details

### 1. Intent Classifier (`intent_classifier.py`)

**Purpose**: Classifies user queries into predefined intent categories to route queries to appropriate retrieval strategies.

**Supported Intents**:
- `flight_search` - Finding flights between locations
- `delay_analysis` - Analyzing flight delays
- `passenger_satisfaction` - Querying satisfaction ratings
- `route_analysis` - Analyzing routes and connections
- `journey_insights` - Getting journey details
- `performance_metrics` - Comparing performance statistics
- `recommendation` - Getting recommendations
- `general_question` - General queries (default)

**How it works**:
- Uses regex patterns to match query text against intent categories
- Scores each intent based on pattern matches
- Returns the intent with the highest score
- Falls back to `general_question` if no matches found

**Example**:
```python
classifier = IntentClassifier()
intent = classifier.classify("Find flights from JFK to LAX")
# Returns: "flight_search"
```

**LLM Enhancement**:
- Optional LLM-based classification for more nuanced understanding
- Falls back to pattern-based if LLM fails

### 2. Entity Extractor (`entity_extractor.py`)

**Purpose**: Extracts structured entities from user queries for precise query construction.

**Supported Entity Types**:
- **AIRPORT**: Airport codes (e.g., "JFK", "LAX") and names (e.g., "New York", "Los Angeles")
- **FLIGHT**: Flight numbers (e.g., "AA123", "DL456")
- **DATE**: Dates in various formats (YYYY-MM-DD, MM/DD/YYYY, month names, years)
- **NUMBER**: Numeric values (integers and floats)

**How it works**:
- Uses regex patterns to identify entities
- Airport codes: 3-letter uppercase codes (IATA format)
- Flight numbers: 2-letter airline code + 3-4 digits
- Dates: Multiple format patterns supported
- Returns a dictionary organized by entity type

**Example**:
```python
extractor = EntityExtractor()
entities = extractor.extract_entities("Find flights from JFK to LAX on 2024-03-15")
# Returns: {
#   "AIRPORT": [
#     {"value": "JFK", "type": "AIRPORT_CODE"},
#     {"value": "LAX", "type": "AIRPORT_CODE"}
#   ],
#   "DATE": [{"value": "2024-03-15", "type": "DATE"}]
# }
```

**LLM Enhancement**:
- Optional LLM-based extraction for complex entity recognition
- Falls back to rule-based extraction if LLM fails

### 3. Embedding Generator (`embedding.py`)

**Purpose**: Generates vector embeddings for semantic similarity search in the knowledge graph.

**Model**: Uses Sentence Transformers (default: `all-MiniLM-L6-v2`)
- Dimension: 384 (default) or 768 (with MPNet)
- Fast and efficient for real-time queries
- Pre-trained on large text corpora

**How it works**:
- Converts text queries into dense vector representations
- Embeddings capture semantic meaning
- Used for similarity search in Neo4j vector index
- Supports single text and batch processing

**Example**:
```python
generator = EmbeddingGenerator()
embedding = generator.embed_text("Find flights from New York to Los Angeles")
# Returns: [0.123, -0.456, 0.789, ...] (384-dimensional vector)
```

**Supported Models**:
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim, fast)
- `sentence-transformers/all-mpnet-base-v2` (768 dim, more accurate)

## Usage in the Application

The preprocessing pipeline is integrated into the main application (`app.py`):

```python
from preprocessing.intent_classifier import IntentClassifier
from preprocessing.entity_extractor import EntityExtractor
from preprocessing.embedding import EmbeddingGenerator

# Initialize components
classifier = IntentClassifier()
extractor = EntityExtractor()
generator = EmbeddingGenerator()

# Process a query
query = "Find flights from JFK to LAX on 2024-03-15"

# Step 1: Classify intent
intent = classifier.classify(query)

# Step 2: Extract entities
entities = extractor.extract_entities(query)

# Step 3: Generate embedding
embedding = generator.embed_text(query)
```

## Testing

### Running Tests

Use the provided test script to verify preprocessing components:

```bash
# Run all tests
python test_preprocessing.py

# Test individual components
python test_preprocessing.py intent      # Test intent classification
python test_preprocessing.py entity      # Test entity extraction
python test_preprocessing.py embedding   # Test embedding generation
python test_preprocessing.py full        # Test complete pipeline

# Interactive mode
python test_preprocessing.py interactive
```

### Expected Output

The test script will:
1. Test intent classification with various query types
2. Test entity extraction with queries containing airports, flights, dates
3. Test embedding generation and show vector dimensions
4. Test the complete pipeline end-to-end

### Example Test Queries

**Intent Classification**:
- "Find flights from JFK to LAX" → `flight_search`
- "What are the delays?" → `delay_analysis`
- "Show me satisfaction ratings" → `passenger_satisfaction`

**Entity Extraction**:
- "Find flights from JFK to LAX on 2024-03-15"
  - Extracts: JFK, LAX (airports), 2024-03-15 (date)
- "What is the status of flight AA123?"
  - Extracts: AA123 (flight number)

**Embedding Generation**:
- Any text query → 384-dimensional vector (or 768 with MPNet)

## Configuration

Preprocessing components use settings from `config.py`:

- **INTENTS**: List of supported intent categories
- **ENTITY_TYPES**: List of supported entity types
- **EMBEDDING_MODELS**: Available embedding models and their dimensions

## Integration with Retrieval

The preprocessing output is used by retrieval components:

1. **Intent** → Routes to appropriate retrieval strategy
2. **Entities** → Used to construct precise Cypher queries
3. **Embeddings** → Used for semantic similarity search in Neo4j

## Future Enhancements

Potential improvements to the preprocessing pipeline:

1. **LLM-based Classification**: More accurate intent detection
2. **NER Models**: Use trained NER models for better entity extraction
3. **Query Expansion**: Expand queries with synonyms and related terms
4. **Multi-language Support**: Support for queries in multiple languages
5. **Contextual Understanding**: Maintain conversation context across queries

## Troubleshooting

### Common Issues

1. **Intent not recognized**: Add more patterns to `intent_patterns` dictionary
2. **Entities not extracted**: Check regex patterns in `EntityExtractor`
3. **Embedding model not loading**: Ensure `sentence-transformers` is installed
4. **Slow embedding generation**: Use smaller model (MiniLM) or enable GPU

### Debug Mode

Enable verbose output by adding print statements in the preprocessing modules or using the interactive test mode.

