"""Test script for input preprocessing pipeline.

This script demonstrates and tests the three main preprocessing components:
1. Intent Classification - Determines user intent from queries
2. Entity Extraction - Extracts structured entities (airports, flights, dates, etc.)
3. Embedding Generation - Creates vector embeddings for semantic search
"""

import json


def test_intent_classifier():
    """Test the IntentClassifier with various queries."""
    from preprocessing.intent_classifier import IntentClassifier
    
    print("=" * 60)
    print("TESTING INTENT CLASSIFIER")
    print("=" * 60)
    
    classifier = IntentClassifier()
    
    test_queries = [
        "Find flights from JFK to LAX",
        "What are the delays for flight AA123?",
        "Show me passenger satisfaction ratings",
        "What routes connect New York to London?",
        "Tell me about passenger journeys",
        "Compare performance metrics between airlines",
        "Recommend the best flight option",
        "How does the airline system work?"
    ]
    
    print("\nQuery Intent Classification Results:\n")
    for query in test_queries:
        intent = classifier.classify(query)
        print(f"Query: {query}")
        print(f"Intent: {intent}")
        print("-" * 60)
    
    print("\n✅ Intent Classifier Test Complete\n")


def test_entity_extractor():
    """Test the EntityExtractor with various queries."""
    from preprocessing.entity_extractor import EntityExtractor
    
    print("=" * 60)
    print("TESTING ENTITY EXTRACTOR")
    print("=" * 60)
    
    extractor = EntityExtractor()
    
    test_queries = [
        "Find flights from JFK to LAX on 2024-03-15",
        "What is the status of flight AA123?",
        "Show me flights from New York to London on January 15, 2024",
        "Find all flights between DFW and ATL with flight number DL456",
        "What are the delays for flights from 2024?",
        "Show me 5 flights from Miami to Chicago",
        # New entity types: Journey, Passenger, Route
        "Show journey details for journey_12345",
        "Find information about journey J56789",
        "What is the status of passenger P12345?",
        "Show passenger passenger_67890 details",
        "What is the route from JFK to LAX?",
        "Tell me about route NYC-LAX"
    ]
    
    print("\nQuery Entity Extraction Results:\n")
    for query in test_queries:
        entities = extractor.extract_entities(query)
        print(f"Query: {query}")
        print(f"Entities: {json.dumps(entities, indent=2)}")
        print("-" * 60)
    
    print("\n✅ Entity Extractor Test Complete\n")


def test_embedding_generator():
    """Test the EmbeddingGenerator."""
    try:
        from preprocessing.embedding import EmbeddingGenerator
    except ImportError as e:
        print("=" * 60)
        print("TESTING EMBEDDING GENERATOR")
        print("=" * 60)
        print(f"\n❌ Error: {e}")
        print("Please install sentence-transformers: pip install sentence-transformers")
        return
    
    print("=" * 60)
    print("TESTING EMBEDDING GENERATOR")
    print("=" * 60)
    
    # Test with default model
    print("\nInitializing embedding model (this may take a moment)...")
    generator = EmbeddingGenerator()
    
    test_texts = [
        "Find flights from New York to Los Angeles",
        "Show me delayed flights",
        "What are passenger satisfaction ratings?",
        "Compare airline performance"
    ]
    
    print(f"\nModel: {generator.model_name}")
    print(f"Embedding Dimension: {generator.get_dimension()}")
    
    print("\nGenerating embeddings for test queries:\n")
    for text in test_texts:
        embedding = generator.embed_text(text)
        print(f"Text: {text}")
        print(f"Embedding shape: {len(embedding)} dimensions")
        print(f"First 5 values: {embedding[:5]}")
        print("-" * 60)
    
    # Test batch embedding
    print("\nTesting batch embedding:")
    batch_embeddings = generator.embed_batch(test_texts)
    print(f"Batch size: {len(batch_embeddings)}")
    print(f"Each embedding dimension: {len(batch_embeddings[0])}")
    
    print("\n✅ Embedding Generator Test Complete\n")


def test_full_pipeline():
    """Test the complete preprocessing pipeline."""
    from preprocessing.intent_classifier import IntentClassifier
    from preprocessing.entity_extractor import EntityExtractor
    
    print("=" * 60)
    print("TESTING FULL PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Initialize all components
    classifier = IntentClassifier()
    extractor = EntityExtractor()
    
    try:
        from preprocessing.embedding import EmbeddingGenerator
        generator = EmbeddingGenerator()
        use_embeddings = True
    except ImportError:
        print("⚠️  EmbeddingGenerator not available (sentence-transformers not installed)")
        print("   Skipping embedding tests in full pipeline\n")
        use_embeddings = False
    
    test_queries = [
        "Find flights from JFK to LAX on 2024-03-15",
        "What are the delays for flight AA123?",
        "Show me passenger satisfaction ratings for flights from New York",
        "Compare performance metrics between different airlines",
        # Test new entity types
        "Show journey details for journey_12345",
        "What is the status of passenger P56789?",
        "Tell me about the route from JFK to LAX"
    ]
    
    print("\nFull Pipeline Results:\n")
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 60)
        
        # Step 1: Intent Classification
        intent = classifier.classify(query)
        print(f"1. Intent: {intent}")
        
        # Step 2: Entity Extraction
        entities = extractor.extract_entities(query)
        print(f"2. Entities: {json.dumps(entities, indent=2)}")
        
        # Step 3: Embedding Generation
        if use_embeddings:
            embedding = generator.embed_text(query)
            print(f"3. Embedding: {len(embedding)} dimensions")
            print(f"   First 5 values: {embedding[:5]}")
        else:
            print(f"3. Embedding: Skipped (sentence-transformers not installed)")
        
        print("\n" + "=" * 60 + "\n")
    
    print("✅ Full Pipeline Test Complete\n")


def test_new_entity_types():
    """Test the newly added entity types: Journey, Passenger, Route."""
    from preprocessing.entity_extractor import EntityExtractor
    
    print("=" * 60)
    print("TESTING NEW ENTITY TYPES (Journey, Passenger, Route)")
    print("=" * 60)
    
    extractor = EntityExtractor()
    
    test_queries = [
        ("Journey ID extraction", [
            "Show journey details for journey_12345",
            "Find information about journey J56789",
            "What happened in journey-98765?",
            "Tell me about journey 11111"
        ]),
        ("Passenger ID extraction", [
            "What is the status of passenger P12345?",
            "Show passenger passenger_67890 details",
            "Find passenger-54321 information",
            "Tell me about passenger 99999"
        ]),
        ("Route extraction", [
            "What is the route from JFK to LAX?",
            "Tell me about route NYC-LAX",
            "Show me the route between New York and Los Angeles",
            "What routes connect JFK to LAX?"
        ])
    ]
    
    print("\nEntity Extraction Results:\n")
    for category, queries in test_queries:
        print(f"\n{category}:")
        print("-" * 60)
        for query in queries:
            entities = extractor.extract_entities(query)
            print(f"Query: {query}")
            print(f"Entities: {json.dumps(entities, indent=2)}")
            print()
    
    print("\n✅ New Entity Types Test Complete\n")


def interactive_test():
    """Interactive mode for testing custom queries."""
    from preprocessing.intent_classifier import IntentClassifier
    from preprocessing.entity_extractor import EntityExtractor
    
    print("=" * 60)
    print("INTERACTIVE PREPROCESSING TEST")
    print("=" * 60)
    print("\nEnter queries to test preprocessing (type 'quit' to exit)\n")
    print("Supported entity types: AIRPORT, FLIGHT, PASSENGER, JOURNEY, ROUTE, DATE, NUMBER\n")
    
    classifier = IntentClassifier()
    extractor = EntityExtractor()
    
    try:
        from preprocessing.embedding import EmbeddingGenerator
        generator = EmbeddingGenerator()
        use_embeddings = True
    except ImportError:
        print("⚠️  EmbeddingGenerator not available (sentence-transformers not installed)")
        use_embeddings = False
    
    while True:
        query = input("Enter a query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        print("\n" + "-" * 60)
        print(f"Query: {query}")
        print("-" * 60)
        
        # Intent
        intent = classifier.classify(query)
        print(f"Intent: {intent}")
        
        # Entities
        entities = extractor.extract_entities(query)
        if entities:
            print(f"Entities: {json.dumps(entities, indent=2)}")
        else:
            print("Entities: None found")
        
        # Embedding (just show dimension)
        if use_embeddings:
            embedding = generator.embed_text(query)
            print(f"Embedding: {len(embedding)} dimensions")
        else:
            print(f"Embedding: Skipped (sentence-transformers not installed)")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "intent":
            test_intent_classifier()
        elif mode == "entity":
            test_entity_extractor()
        elif mode == "embedding":
            test_embedding_generator()
        elif mode == "full":
            test_full_pipeline()
        elif mode == "new-entities":
            test_new_entity_types()
        elif mode == "interactive":
            interactive_test()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: intent, entity, embedding, full, new-entities, interactive")
    else:
        # Run all tests
        test_intent_classifier()
        test_entity_extractor()
        test_embedding_generator()
        test_new_entity_types()
        test_full_pipeline()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETE")
        print("=" * 60)
        print("\nTo run interactive mode: python test_preprocessing.py interactive")
        print("To test individual components:")
        print("  python test_preprocessing.py intent")
        print("  python test_preprocessing.py entity")
        print("  python test_preprocessing.py embedding")
        print("  python test_preprocessing.py new-entities  # Test Journey, Passenger, Route")
        print("  python test_preprocessing.py full")

