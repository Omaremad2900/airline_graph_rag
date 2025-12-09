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
- Scores each intent based on the number of matching patterns
- Returns the intent with the highest score
- Falls back to `general_question` if no matches found
- Case-insensitive matching for robust classification

**Example**:
```python
classifier = IntentClassifier()
intent = classifier.classify("Find flights from JFK to LAX")
# Returns: "flight_search"
```

**LLM Enhancement**:
- Optional LLM-based classification via `classify_with_llm()` method
- Takes an optional LLM instance parameter
- Uses structured prompt with available intents from config
- Falls back to pattern-based classification if LLM fails or is not provided
- Validates LLM response against configured intent list

### 2. Entity Extractor (`entity_extractor.py`)

**Purpose**: Extracts structured entities from user queries for precise query construction.

**Supported Entity Types**:
- **AIRPORT**: Airport codes (e.g., "JFK", "LAX") and names (e.g., "New York", "Los Angeles")
- **FLIGHT**: Flight numbers (e.g., "AA123", "DL456")
- **DATE**: Dates in various formats (YYYY-MM-DD, MM/DD/YYYY, month names, years)
- **NUMBER**: Numeric values (integers and floats)

**How it works**:
- Uses regex patterns to identify entities with intelligent filtering
- **Airport Codes**: 
  - 3-letter uppercase codes (IATA format)
  - Uses a whitelist of valid airport codes to prevent false positives
  - Excludes common 3-letter words (e.g., "THE", "AND", "FOR")
  - Supports major US and international airports
- **Airport Names**: 
  - Recognizes common airport names and city names
  - Maps airport names to their corresponding codes
  - Deduplicates: if an airport is found as both code and name, keeps only the code version
- **Flight Numbers**: 2-letter airline code + 3-4 digits (e.g., "AA123", "DL456")
- **Dates**: 
  - Multiple format patterns supported (YYYY-MM-DD, MM/DD/YYYY, month names)
  - Handles year-only extraction (if not part of a full date)
  - Prioritizes full dates over year-only matches
- **Numbers**: 
  - Extracts numeric values (integers and floats)
  - Automatically excludes numbers that are part of date entities
  - Prevents duplicate extraction of date components
- Returns a dictionary organized by entity type (only includes non-empty entity lists)

**Key Features**:
- **Deduplication**: Prevents duplicate entities (e.g., JFK found as both code and "New York" name)
- **False Positive Prevention**: Whitelist-based airport code validation
- **Smart Number Extraction**: Excludes numbers that are part of dates
- **Separate Extraction Methods**: Individual methods for each entity type (`extract_airports()`, `extract_flights()`, `extract_dates()`, `extract_numbers()`)

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

# Example with airport name
entities = extractor.extract_entities("Find flights from New York to Los Angeles")
# Returns: {
#   "AIRPORT": [
#     {"value": "jfk", "type": "AIRPORT_NAME"},  # or mapped to "JFK" if in mapping
#     {"value": "lax", "type": "AIRPORT_NAME"}
#   ]
# }
```

**LLM Enhancement**:
- Optional LLM-based extraction via `extract_with_llm()` method
- Takes an optional LLM instance parameter
- Falls back to rule-based extraction if LLM fails or is not provided

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
- Automatically determines embedding dimension from model configuration

**Methods**:
- `embed_text(text)`: Generate embedding for a single text string
- `embed_batch(texts)`: Generate embeddings for multiple texts efficiently
- `get_dimension()`: Get the dimension of embeddings for the current model

**Example**:
```python
generator = EmbeddingGenerator()
embedding = generator.embed_text("Find flights from New York to Los Angeles")
# Returns: [0.123, -0.456, 0.789, ...] (384-dimensional vector)

# Batch processing
texts = ["Find flights", "Show delays", "Compare airlines"]
embeddings = generator.embed_batch(texts)
# Returns: [[...], [...], [...]] (list of embedding vectors)

# Get dimension
dim = generator.get_dimension()  # Returns: 384 (for MiniLM) or 768 (for MPNet)
```

**Supported Models**:
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim, fast) - **Default**
- `sentence-transformers/all-mpnet-base-v2` (768 dim, more accurate)

**Customization**:
- Can specify model name in constructor: `EmbeddingGenerator(model_name="sentence-transformers/all-mpnet-base-v2")`
- Model configuration (dimensions) defined in `config.py` under `EMBEDDING_MODELS`

## Usage in the Application

The preprocessing pipeline is integrated into the main application (`app.py`):

```python
from preprocessing.intent_classifier import IntentClassifier
from preprocessing.entity_extractor import EntityExtractor
from preprocessing.embedding import EmbeddingGenerator

# Initialize components
classifier = IntentClassifier()
extractor = EntityExtractor()
generator = EmbeddingGenerator()  # Optional: specify model_name parameter

# Process a query
query = "Find flights from JFK to LAX on 2024-03-15"

# Step 1: Classify intent
intent = classifier.classify(query)
# Or use LLM-based: intent = classifier.classify_with_llm(query, llm=llm_instance)

# Step 2: Extract entities
entities = extractor.extract_entities(query)
# Or use LLM-based: entities = extractor.extract_with_llm(query, llm=llm_instance)

# Step 3: Generate embedding (optional, used for semantic search)
embedding = generator.embed_text(query)
# Or batch: embeddings = generator.embed_batch([query1, query2, ...])
```

**In Streamlit App**:
- Components are initialized in session state for performance
- Preprocessing results are displayed in an expandable section
- Intent and entities are used to route queries to appropriate retrieval strategies
- Embeddings are used for semantic similarity search when embedding-based retrieval is selected

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
- "Compare performance metrics" → `performance_metrics`
- "Recommend the best flight" → `recommendation`

**Entity Extraction**:
- "Find flights from JFK to LAX on 2024-03-15"
  - Extracts: JFK, LAX (airports as codes), 2024-03-15 (date)
- "What is the status of flight AA123?"
  - Extracts: AA123 (flight number)
- "Show me flights from New York to London on January 15, 2024"
  - Extracts: New York, London (airport names), January 15, 2024 (date)
- "Find all flights between DFW and ATL with flight number DL456"
  - Extracts: DFW, ATL (airports), DL456 (flight number)
- "Show me 5 flights from Miami to Chicago"
  - Extracts: Miami, Chicago (airport names), 5 (number, not part of date)

**Embedding Generation**:
- Any text query → 384-dimensional vector (default MiniLM) or 768-dimensional (MPNet)
- Batch processing supported for multiple queries

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

## Advanced Features

### Entity Extractor Advanced Capabilities

**Airport Code Whitelist**:
- The system maintains a whitelist of valid IATA airport codes
- Prevents false positives from common 3-letter words
- Includes major US and international airports
- Can be extended by adding codes to `valid_airport_codes` set

**Deduplication Logic**:
- Automatically handles cases where an airport is found as both code and name
- Prefers code version over name version
- Example: "JFK" and "New York" both found → keeps only "JFK" as AIRPORT_CODE

**Smart Number Extraction**:
- Extracts dates first to identify date components
- Excludes numbers that are part of date entities
- Prevents duplicate extraction (e.g., "2024" in "2024-03-15" won't be extracted as NUMBER)

**Separate Extraction Methods**:
- `extract_airports(query)`: Extract only airport entities
- `extract_flights(query)`: Extract only flight numbers
- `extract_dates(query)`: Extract only date entities
- `extract_numbers(query, exclude_dates)`: Extract only numeric values
- Useful for targeted extraction or debugging

### Embedding Generator Advanced Features

**Batch Processing**:
- `embed_batch(texts)` efficiently processes multiple texts
- Faster than calling `embed_text()` multiple times
- Useful for processing multiple queries or documents

**Model Configuration**:
- Model dimensions are automatically read from `config.py`
- Supports easy switching between models
- Dimension information available via `get_dimension()` method

## Future Enhancements

Potential improvements to the preprocessing pipeline:

1. **LLM-based Classification**: More accurate intent detection (partially implemented)
2. **NER Models**: Use trained NER models for better entity extraction (partially implemented)
3. **Query Expansion**: Expand queries with synonyms and related terms
4. **Multi-language Support**: Support for queries in multiple languages
5. **Contextual Understanding**: Maintain conversation context across queries
6. **Entity Linking**: Link extracted entities to knowledge graph nodes
7. **Confidence Scores**: Add confidence scores to intent and entity extraction

## Troubleshooting

### Common Issues

1. **Intent not recognized**: 
   - Add more patterns to `intent_patterns` dictionary in `IntentClassifier`
   - Patterns are case-insensitive and use regex matching
   - Consider using LLM-based classification for complex queries

2. **Entities not extracted**: 
   - Check if airport code is in the whitelist (`valid_airport_codes`)
   - Verify regex patterns in `EntityExtractor`
   - Airport codes must be 3-letter uppercase IATA codes
   - Common words are excluded to prevent false positives

3. **False positive airport codes**: 
   - The system uses a whitelist to prevent matching common 3-letter words
   - If a valid airport code is not recognized, add it to `valid_airport_codes` set

4. **Numbers extracted as dates**: 
   - The system automatically excludes numbers that are part of dates
   - Date extraction happens first, then numbers are filtered

5. **Embedding model not loading**: 
   - Ensure `sentence-transformers` is installed: `pip install sentence-transformers`
   - First run will download the model (may take time)
   - Check internet connection for model download

6. **Slow embedding generation**: 
   - Use smaller model (MiniLM-L6-v2, 384 dim) instead of MPNet (768 dim)
   - Enable GPU if available (automatic if CUDA is installed)
   - Use batch processing for multiple queries

7. **Duplicate entities**: 
   - The system automatically deduplicates airports found as both code and name
   - Code version is preferred over name version

### Debug Mode

Enable verbose output by:
- Using the interactive test mode: `python test_preprocessing.py interactive`
- Adding print statements in the preprocessing modules
- Checking the preprocessing results in the Streamlit app's expandable section


