# How Preprocessing Works: Entity Extraction, Intent Classification, and Embeddings

This document explains how the three main preprocessing components work in detail.

---

## 1. Intent Classification

### Purpose
**Intent Classification** determines what the user wants to do with their query. It routes queries to the appropriate retrieval strategies.

### How It Works

#### **Step 1: Pattern Matching (Rule-Based)**
The system uses **regex patterns** to match keywords and phrases in the query:

```python
intent_patterns = {
    "flight_search": [
        r"find.*flight",
        r"search.*flight", 
        r"flights?.*from.*to"
    ],
    "delay_analysis": [
        r"delay",
        r"late",
        r"on.?time"
    ],
    "passenger_satisfaction": [
        r"satisfaction",
        r"rating",
        r"feedback"
    ],
    # ... 5 more intent categories
}
```

#### **Step 2: Scoring System**
- Each intent has multiple patterns
- The system counts how many patterns match the query
- **Example**: Query "find flights from JFK to LAX"
  - `flight_search`: 3 matches (find.*flight, search.*flight, flights?.*from.*to)
  - `delay_analysis`: 0 matches
  - `passenger_satisfaction`: 0 matches
  - **Winner**: `flight_search` (highest score)

#### **Step 3: Classification Result**
```python
def classify(query: str) -> str:
    # Score each intent
    intent_scores = {}
    for intent, patterns in self.intent_patterns.items():
        score = sum(1 for pattern in patterns 
                   if re.search(pattern, query.lower()))
        if score > 0:
            intent_scores[intent] = score
    
    # Return highest scoring intent
    return max(intent_scores, key=intent_scores.get) 
    # or "general_question" if no matches
```

#### **Step 4: LLM Enhancement (Optional)**
If an LLM is available, the system can use it for better classification:
```python
def classify_with_llm(query: str, llm) -> str:
    prompt = f"""Classify this query into one of: {INTENTS}
    Query: {query}
    Respond with only the intent name."""
    
    response = llm.invoke(prompt)
    # Falls back to rule-based if LLM fails
```

### Example Flow
```
Query: "What flights are delayed from New York?"
↓
Pattern Matching:
  - "delay" → delay_analysis: +1
  - "flights" → flight_search: +1
  - "from" → flight_search: +1
↓
Scores: delay_analysis=1, flight_search=2
↓
Result: "flight_search" (highest score)
```

---

## 2. Entity Extraction

### Purpose
**Entity Extraction** identifies structured information (airports, flights, dates, etc.) from the query to fill Cypher query parameters.

### How It Works

The system extracts **7 types of entities**:

#### **A. Airport Extraction**

**Method**: Regex + Whitelist Filtering

```python
# Step 1: Find all 3-letter uppercase codes
airport_code_pattern = r'\b[A-Z]{3}\b'
codes = re.findall(pattern, "JFK LAX ORD")  # Finds: JFK, LAX, ORD

# Step 2: Filter by whitelist (prevents false positives)
valid_airport_codes = {"JFK", "LAX", "ORD", "DFW", ...}
excluded_words = {"THE", "AND", "FOR", "ARE", ...}  # Common words

# Step 3: Only keep valid airports
for code in codes:
    if code in valid_airport_codes and code not in excluded_words:
        airports.append(code)
```

**Smart Features**:
- **Deduplication**: If "JFK" and "John F. Kennedy" both appear, keeps only "JFK"
- **Name Mapping**: "New York" → "JFK" (if configured)
- **False Positive Prevention**: Excludes common 3-letter words like "THE", "AND"

**Example**:
```
Query: "Flights from JFK to LAX"
Extracted: [
    {"value": "JFK", "type": "AIRPORT_CODE"},
    {"value": "LAX", "type": "AIRPORT_CODE"}
]
```

#### **B. Flight Number Extraction**

**Method**: Regex Pattern Matching

```python
flight_number_pattern = r'\b[A-Z]{2}\d{3,4}\b'
# Matches: AA123, DL4567, UA1234
```

**Example**:
```
Query: "Show me details for flight AA123"
Extracted: [{"value": "AA123", "type": "FLIGHT"}]
```

#### **C. Journey ID Extraction**

**Method**: Regex with Explicit Prefix Requirement

```python
journey_id_pattern = r'\b(?:(?:journey[_-]|j[_-])(\d{4,})|j(\d{4,}))\b'
# Matches: journey_12345, journey-12345, J12345, j_12345
# Does NOT match: 12345 (to avoid matching years)
```

**Why the prefix?** Prevents false positives:
- ❌ Without prefix: "2024" would match as journey ID
- ✅ With prefix: Only "journey_12345" or "J12345" matches

**Example**:
```
Query: "Show journey_12345 details"
Extracted: [{"value": "12345", "type": "JOURNEY"}]
```

#### **D. Passenger ID Extraction**

**Method**: Similar to Journey IDs

```python
passenger_id_pattern = r'\b(?:(?:passenger[_-]|p[_-])(\d{4,})|p(\d{4,}))\b'
# Matches: passenger_12345, passenger-12345, P12345, p_12345
```

**Example**:
```
Query: "What did passenger P12345 say?"
Extracted: [{"value": "12345", "type": "PASSENGER"}]
```

#### **E. Route Extraction**

**Method**: Pattern Matching for Route Mentions

```python
route_pattern = r'\broute[_-]?(\w+)?\b'
# Extracts route mentions (routes are typically implicit via airport pairs)
```

**Example**:
```
Query: "Analyze routes between major airports"
Extracted: [{"value": "mentioned", "type": "ROUTE"}]
```

#### **F. Date Extraction**

**Method**: Multiple Pattern Matching (Most Specific First)

```python
date_patterns = [
    r'\b\d{4}-\d{2}-\d{2}\b',           # 2024-01-15
    r'\b\d{1,2}/\d{1,2}/\d{4}\b',        # 01/15/2024
    r'\b(?:january|february|...)\s+\d{1,2},?\s+\d{4}\b'  # January 15, 2024
]
```

**Smart Feature**: Prevents double extraction
- Full date "2024-01-15" → extracts as one DATE
- Year "2024" alone → extracts separately (if not part of full date)

**Example**:
```
Query: "Flights on 2024-01-15"
Extracted: [{"value": "2024-01-15", "type": "DATE"}]
```

#### **G. Number Extraction**

**Method**: Regex with Smart Exclusion

```python
number_pattern = r'\b\d+(?:\.\d+)?\b'
# Matches: 123, 45.67, 1000
```

**Smart Features**:
1. **Excludes numbers in dates**: "2024-01-15" → doesn't extract "2024", "01", "15"
2. **Excludes entity IDs**: If "journey_12345" found, doesn't extract "12345" as NUMBER
3. **Position-based filtering**: Checks if number overlaps with date/ID positions

**Example**:
```
Query: "Show top 10 flights with rating above 4.5"
Extracted: [
    {"value": 10.0, "type": "NUMBER"},
    {"value": 4.5, "type": "NUMBER"}
]
# Note: If query had "journey_12345", "12345" is NOT extracted as NUMBER
```

### Complete Extraction Flow

```python
def extract_entities(query: str) -> dict:
    # Step 1: Extract dates first (needed for number exclusion)
    dates = extract_dates(query)
    
    # Step 2: Extract other entities
    entities = {
        "AIRPORT": extract_airports(query),
        "FLIGHT": extract_flights(query),
        "PASSENGER": extract_passengers(query),
        "JOURNEY": extract_journeys(query),
        "ROUTE": extract_routes(query),
        "DATE": dates,
        "NUMBER": extract_numbers(query, exclude_dates=dates)
    }
    
    # Step 3: Filter out empty lists
    return {k: v for k, v in entities.items() if v}
```

### Example: Full Query

```
Query: "Find flights from JFK to LAX on 2024-01-15 for passenger P12345"

Extracted Entities:
{
    "AIRPORT": [
        {"value": "JFK", "type": "AIRPORT_CODE"},
        {"value": "LAX", "type": "AIRPORT_CODE"}
    ],
    "DATE": [
        {"value": "2024-01-15", "type": "DATE"}
    ],
    "PASSENGER": [
        {"value": "12345", "type": "PASSENGER"}
    ]
}
```

---

## 3. Embeddings

### Purpose
**Embeddings** convert text into numerical vectors that capture semantic meaning. Used for **semantic similarity search** in the knowledge graph.

### How It Works

#### **Step 1: Model Selection**

The system uses **Sentence Transformers** models:

```python
EMBEDDING_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "name": "MiniLM-L6-v2",
        "dimension": 384,  # Output vector size
        "type": "sentence-transformers"
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "name": "MPNet-Base-v2", 
        "dimension": 768,  # Larger, more accurate
        "type": "sentence-transformers"
    }
}
```

#### **Step 2: Model Loading**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# Downloads model on first use (if not cached)
# Loads into memory (~80MB for MiniLM)
```

#### **Step 3: Text Encoding**

```python
def embed_text(text: str) -> list:
    # Convert text to vector
    embedding = model.encode(text, convert_to_numpy=True)
    # Returns: [0.123, -0.456, 0.789, ...] (384 numbers for MiniLM)
    return embedding.tolist()
```

**What happens internally**:
1. Tokenization: Text → tokens
2. Model processing: Tokens → hidden states
3. Pooling: Hidden states → single vector (384 or 768 dimensions)
4. Normalization: Vector is normalized (optional)

#### **Step 4: Batch Processing**

```python
def embed_batch(texts: list) -> list:
    # Process multiple texts efficiently
    embeddings = model.encode(texts, convert_to_numpy=True)
    # Returns: [[vec1], [vec2], [vec3], ...]
    return embeddings.tolist()
```

### When Embeddings Are Used

#### **1. Knowledge Graph Initialization** (One-time setup)
```python
# For each journey/feedback in Neo4j:
feature_text = f"Journey {journey_id}: {feedback_text} ..."
embedding = embedding_model.embed_text(feature_text)
# Store in FAISS vector database
```

#### **2. Query Processing** (Per query)
```python
# User query: "What are passengers saying about delays?"
query_embedding = embedding_model.embed_text(user_query)
# Search FAISS for similar embeddings
# Returns top-k most similar journeys
```

### How Similarity Works

**Cosine Similarity** (most common):
```
similarity = cosine(query_vector, stored_vector)
# Range: -1 to 1
# 1.0 = identical meaning
# 0.0 = unrelated
# -1.0 = opposite meaning
```

**Example**:
```
Query: "delayed flights"
Embedding: [0.1, -0.3, 0.5, ...]

Stored Text 1: "flight was delayed by 2 hours"
Embedding: [0.12, -0.28, 0.48, ...]
Similarity: 0.95 (very similar)

Stored Text 2: "excellent food service"
Embedding: [-0.2, 0.4, -0.1, ...]
Similarity: 0.15 (not similar)
```

### Important: Model Consistency

**Critical Rule**: The same embedding model must be used for:
1. **Knowledge Graph embeddings** (stored in FAISS)
2. **Query embeddings** (user input)

**Why?** Different models produce different vector spaces. You can't compare embeddings from different models!

```python
# ✅ CORRECT: Same model
kg_embedding = model_A.embed_text("journey data")
query_embedding = model_A.embed_text("user query")
similarity = cosine(kg_embedding, query_embedding)  # Works!

# ❌ WRONG: Different models
kg_embedding = model_A.embed_text("journey data")
query_embedding = model_B.embed_text("user query")
similarity = cosine(kg_embedding, query_embedding)  # Meaningless!
```

---

## Complete Preprocessing Pipeline Flow

```
User Query: "Find delayed flights from JFK to LAX on 2024-01-15"
↓
┌─────────────────────────────────────────────────┐
│ 1. Intent Classification                        │
│    Pattern matching → Scoring → "flight_search" │
└─────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────┐
│ 2. Entity Extraction                            │
│    AIRPORT: ["JFK", "LAX"]                      │
│    DATE: ["2024-01-15"]                         │
│    (delay keyword → intent, not entity)         │
└─────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────┐
│ 3. Embedding Generation (if using embeddings)   │
│    Query → [0.123, -0.456, ...] (384-dim)      │
│    Used for semantic search in FAISS            │
└─────────────────────────────────────────────────┘
↓
Output to Retrieval Layer:
{
    "intent": "flight_search",
    "entities": {
        "AIRPORT": ["JFK", "LAX"],
        "DATE": ["2024-01-15"]
    },
    "embedding": [0.123, -0.456, ...]  # Optional
}
```

---

## Error Handling & Validation

All three components include robust error handling:

### Intent Classification
- ✅ Validates input (None, empty, non-string)
- ✅ Handles regex errors gracefully
- ✅ Falls back to "general_question" on errors
- ✅ Logs warnings/errors

### Entity Extraction
- ✅ Validates input before processing
- ✅ Whitelist filtering prevents false positives
- ✅ Smart exclusion (dates vs numbers, IDs vs numbers)
- ✅ Returns empty dict on errors (safe fallback)

### Embeddings
- ✅ Validates model loading
- ✅ Handles encoding errors
- ✅ Returns zero vector on failure (safe fallback)
- ✅ Validates input text (None, empty, non-string)

---

## Summary

| Component | Method | Output | Used For |
|-----------|--------|--------|----------|
| **Intent Classification** | Regex patterns + scoring | Intent string | Routing to retrieval strategy |
| **Entity Extraction** | Regex + whitelist + smart filtering | Dict of entities | Filling Cypher query parameters |
| **Embeddings** | Sentence Transformers | Vector (384/768 dim) | Semantic similarity search in FAISS |

All three work together to prepare user queries for efficient retrieval from the Neo4j knowledge graph!

