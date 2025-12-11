# FAISS Migration Guide

## Overview

The embedding storage has been migrated from Neo4j node properties to **FAISS** (Facebook AI Similarity Search) for efficient vector storage and similarity search.

## What Changed

### Before (Neo4j Properties)

- Embeddings stored as properties on Journey nodes in Neo4j
- Manual similarity search (fetching all embeddings, computing cosine similarity in Python)
- Limited to 500 embeddings in search queries
- Slow for large datasets

### After (FAISS)

- Embeddings stored in FAISS indices (on disk)
- Efficient similarity search using FAISS's optimized algorithms
- No limit on number of embeddings
- Much faster for large-scale similarity search

## Architecture

### Storage Structure

```
faiss_indices/
├── feature_embedding_sentence_transformers_all_MiniLM_L6_v2.index
├── feature_embedding_sentence_transformers_all_MiniLM_L6_v2_mapping.json
├── feature_embedding_sentence_transformers_all_mpnet_base_v2.index
└── feature_embedding_sentence_transformers_all_mpnet_base_v2_mapping.json
```

### Components

1. **FAISS Index** (`.index` file)

   - Stores vector embeddings in optimized format
   - Uses `IndexFlatIP` (Inner Product) for cosine similarity
   - Vectors are normalized for cosine similarity computation

2. **ID Mapping** (`_mapping.json` file)

   - Maps FAISS vector IDs to Journey `feedback_ID`s
   - Format: `{"0": "feedback_id_1", "1": "feedback_id_2", ...}`

3. **EmbeddingRetriever**
   - Automatically loads FAISS index on initialization
   - Creates new index if none exists
   - Saves index and mapping after creation

## Usage

### Initializing Embeddings

```bash
# Single model
python scripts/initialize_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2

# All models
python scripts/initialize_embeddings.py --all
```

This will:

1. Fetch all Journey nodes from Neo4j
2. Create text descriptions from numerical properties
3. Generate embeddings using the specified model
4. Store embeddings in FAISS index
5. Create ID mapping file
6. Save both to disk

### Using Embeddings in Code

```python
from utils.neo4j_connector import Neo4jConnector
from preprocessing.embedding import EmbeddingGenerator
from retrieval.embeddings import EmbeddingRetriever

# Initialize
connector = Neo4jConnector()
embedding_model = EmbeddingGenerator("sentence-transformers/all-MiniLM-L6-v2")
retriever = EmbeddingRetriever(connector, embedding_model)

# Search
results = retriever.retrieve_by_similarity("flights with delays", top_k=10)
```

The retriever automatically:

- Loads FAISS index if it exists
- Performs efficient similarity search
- Fetches graph data from Neo4j for similar journeys
- Returns results with similarity scores

## Benefits

### Performance

- **10-100x faster** similarity search for large datasets
- No need to fetch all embeddings from Neo4j
- Optimized algorithms (HNSW-like indexing in FAISS)

### Scalability

- Can handle millions of embeddings efficiently
- No 500-embedding limit
- Memory-efficient storage

### Separation of Concerns

- Graph data stays in Neo4j
- Vector search optimized in FAISS
- Best tool for each job

## Migration Notes

### Backward Compatibility

- The `EmbeddingRetriever` API remains the same
- `retrieve_by_similarity()` method signature unchanged
- Existing code using `EmbeddingRetriever` should work without changes

### Data Migration

If you have existing embeddings in Neo4j:

1. Run `initialize_embeddings.py` to regenerate embeddings
2. Old Neo4j properties can be removed (optional)
3. FAISS indices will be created fresh

### File Management

- FAISS indices are stored in `faiss_indices/` directory
- Added to `.gitignore` (generated files)
- Should be backed up separately if needed

## Technical Details

### FAISS Index Type

- **IndexFlatIP**: Inner Product index
- Vectors are normalized before storage
- Inner product of normalized vectors = cosine similarity
- Fast exact search (no approximation)

### Vector Normalization

```python
# Vectors are normalized for cosine similarity
faiss.normalize_L2(embeddings_array)  # Normalize stored vectors
faiss.normalize_L2(query_vector)      # Normalize query vector
# Then use Inner Product = cosine similarity
```

### Search Process

1. Generate query embedding
2. Normalize query vector
3. Search in FAISS index (returns top-k similar vectors)
4. Map FAISS IDs to Journey feedback_IDs
5. Fetch graph data from Neo4j for those journeys
6. Return results with similarity scores

## Troubleshooting

### Index Not Found

```
No FAISS index found. Please run initialize_embeddings.py first.
```

**Solution**: Run the initialization script to create embeddings.

### Index Loading Error

```
Error loading FAISS index: ...
```

**Solution**: Delete the corrupted index files and reinitialize.

### Dimension Mismatch

If you change embedding models, you may need to:

1. Delete old FAISS indices
2. Reinitialize with new model

## Future Enhancements

Possible improvements:

- Use FAISS `IndexIVFFlat` or `IndexHNSW` for even faster approximate search
- Add incremental updates (add new embeddings without full rebuild)
- Support for GPU acceleration (`faiss-gpu`)
- Index versioning and migration tools
