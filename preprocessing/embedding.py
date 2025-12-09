"""Embedding generation for semantic similarity search."""
from sentence_transformers import SentenceTransformer
import numpy as np
import config


class EmbeddingGenerator:
    """Generates embeddings for text inputs."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = config.EMBEDDING_MODELS.get(model_name, {}).get("dimension", 384)
    
    def embed_text(self, text: str) -> list:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as a list
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: list) -> list:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.dimension

