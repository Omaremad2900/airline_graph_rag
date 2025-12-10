# Graph-RAG Airline Assistant: Concise Presentation Summary

## 1. System Architecture

### Overview
End-to-end system combining Neo4j Knowledge Graph with LLMs. Four-layer pipeline: **Preprocessing → Retrieval → LLM → UI**

### Architecture Flow
```
User Query → Intent Classification + Entity Extraction → 
Cypher Queries + Embedding Search → 
LLM Generation → Response Display
```

### Components

**1. Input Preprocessing**
- **Intent Classification**: 8 categories (flight_search, delay_analysis, satisfaction, etc.)
- **Entity Extraction**: 7 types (Airport, Flight, Passenger, Journey, Route, Date, Number)
- **Embedding Generation**: 2 models (MiniLM-L6-v2: 384 dim, MPNet-Base-v2: 768 dim)

**2. Graph Retrieval**
- **Baseline**: 18 Cypher query templates (exceeds 10+ requirement)
- **Embeddings**: Feature vector embeddings with semantic similarity search
- **Hybrid**: Combines both methods

**3. LLM Layer**
- **5+ Providers**: OpenAI, Anthropic, Google, OpenRouter, HuggingFace
- **10+ Models**: GPT-3.5, GPT-4, Claude 3 Haiku, Gemini Pro, Mistral, Llama
- **Structured Prompts**: Persona + Task + Context + Query

**4. User Interface**
- Streamlit app with query transparency, graph visualization, model comparison

---

## 2. Retrieval Strategy and Examples

### Baseline Retrieval (Cypher Queries)

**18 Query Templates by Intent:**
- Flight Search: 3 (by route, departure, arrival)
- Delay Analysis: 3 (delays above threshold, by route, worst delays)
- Passenger Satisfaction: 3 (low ratings, by class, poor performance)
- Route Analysis: 3 (popular routes, performance, multi-leg)
- Journey Insights: 2 (details, loyalty passengers)
- Performance Metrics: 2 (overall stats, flight performance)
- Recommendation: 1 (best routes)
- General Question: 1 (flight info)

**Example:**
- Query: "Find flights from JFK to LAX"
- Intent: `flight_search`
- Entities: `{AIRPORT: [JFK, LAX]}`
- Cypher: `by_route` template with parameters
- Results: List of matching flights

### Embedding-Based Retrieval

**Feature Vector Embeddings:**
- Constructs text from Journey properties: "Journey ABC123. from ORD to LAX. Flight AA123. Class Economy. Food satisfaction 4. Arrived on time."
- Two models: MiniLM-L6-v2 (fast) and MPNet-Base-v2 (accurate)
- Cosine similarity search

**Example:**
- Query: "Show journeys with good food and minimal delays"
- Process: Embed query → Find similar journeys → Rank by similarity
- Results: Top-k journeys with high food satisfaction and low delays

### Hybrid Retrieval

**Example:** "Flights from Chicago with delays"
- Baseline: Executes `by_departure` + `flights_with_delays`
- Embeddings: Semantic search for "delays" and "Chicago"
- Results: Combined and deduplicated

---

## 3. LLM Comparison

### Quantitative Metrics

| Model | Response Time | Tokens | Length |
|-------|--------------|--------|--------|
| GPT-3.5-turbo | 2.3s | 450 | 1,250 |
| GPT-4 | 4.1s | 520 | 1,380 |
| Claude 3 Haiku | 3.2s | 480 | 1,320 |
| Gemini Pro | 2.8s | 460 | 1,290 |

### Qualitative Analysis

- **GPT-3.5-turbo**: Fast, concise, accurate
- **GPT-4**: Most detailed, highest accuracy, slower
- **Claude 3 Haiku**: Balanced performance, good naturalness
- **Gemini Pro**: Fast, good accuracy

**Recommendations:**
- Speed: GPT-3.5-turbo or Gemini Pro
- Accuracy: GPT-4
- Balance: Claude 3 Haiku

---

## 4. Error Analysis

### Intent Classification Errors
- **Ambiguous queries**: ~10% misclassification
- **Unrecognized**: ~5% (falls back to general_question)
- **Examples**: "Show me flights" → Could be flight_search or general_question

### Entity Extraction Errors
- **False positives**: ~2% (airport codes like "THE" mistaken)
- **False negatives**: ~8% (codes not in whitelist)
- **Missing entities**: ~5%
- **Examples**: "Find flights from THE airport" → "THE" incorrectly extracted

### Retrieval & LLM Errors
- **No results**: Query parameters don't match data
- **Hallucination**: LLM generates information not in context
- **Incomplete answers**: Missing relevant information
- **API failures**: Network issues or rate limits

### Error Rates Summary
- Intent Classification: ~5-10%
- Entity Extraction: ~2-8%
- Embedding Generation: ~1%

---

## 5. Improvements Added

### Preprocessing
1. ✅ **Airport Code Recognition**: Expanded whitelist to 50+ airports
2. ✅ **Intent Scoring**: Changed from first-match to scoring system
3. ✅ **Date/Number Resolution**: Eliminated duplicate extraction
4. ✅ **Entity Extraction**: Added Journey/Passenger/Route entities
5. ✅ **False Positive Prevention**: Prefix requirement for entity IDs

### Retrieval
1. ✅ **Hybrid Strategy**: Combined Cypher + semantic search
2. ✅ **Multiple Embedding Models**: Support for 2 models
3. ✅ **Feature Vector Embeddings**: Text descriptions from numerical properties

### LLM & UI
1. ✅ **Structured Prompts**: Persona + Task + Context format
2. ✅ **Multi-Model Support**: 5+ LLM providers
3. ✅ **Metrics Tracking**: Response time, tokens, length
4. ✅ **Query Transparency**: Display executed Cypher queries
5. ✅ **Model Comparison**: Side-by-side comparison
6. ✅ **Graph Visualization**: Interactive visualization

**Results**: Error rates reduced significantly, better functionality across all layers

---

## 6. Remaining Limitations

### Preprocessing
- Multi-intent queries not supported
- Airport codes not in whitelist need manual addition
- Complex date formats (e.g., "next Monday") not supported
- ~10% misclassification, ~8% false negatives

### Retrieval
- Baseline: Structured queries only, exact matching
- Embeddings: Quality depends on feature text, computational cost
- Hybrid: Simple deduplication, no weighted combination

### LLM & System
- Hallucination risk, incomplete answers
- Missing data cannot be answered
- Context size limits (top 30 records)
- Limited scalability for very large graphs
- API dependencies and costs

### Future Improvements
1. Machine learning models for NER
2. Intent confidence scores
3. Vector index optimization
4. Context awareness
5. Multi-language support
6. Feedback loop mechanism

---

## Summary Statistics

**System Metrics:**
- Query Templates: **18** (exceeds 10+ requirement)
- Intent Categories: **8**
- Entity Types: **7**
- Embedding Models: **2**
- LLM Providers: **5+**
- Supported Models: **10+**

**Performance:**
- Baseline Query: < 1 second
- Embedding Search: 1-3 seconds
- LLM Response: 2-5 seconds
- Total Pipeline: 3-9 seconds

**Error Rates:**
- Intent Classification: ~5-10%
- Entity Extraction: ~2-8%
- Embedding Generation: ~1%

---

## Conclusion

✅ **Comprehensive Architecture**: 4-layer pipeline with clear separation
✅ **Dual Retrieval**: Baseline Cypher + Embedding-based semantic search
✅ **Multi-Model LLM**: 5+ providers with quantitative comparison
✅ **Error Handling**: Graceful degradation and improvements
✅ **Continuous Improvement**: Multiple enhancements addressing limitations

**Future Work**: Machine learning enhancements, context awareness, scalability improvements

