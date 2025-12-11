"""Embedding generation for semantic similarity search."""
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import config

# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for text inputs."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the embedding model to use
            
        Raises:
            ValueError: If model_name is empty or invalid
            RuntimeError: If model loading fails
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")
        
        self.model_name = model_name
        
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dimension = config.EMBEDDING_MODELS.get(model_name, {}).get("dimension", 384)
            logger.info(f"Successfully loaded model with dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{model_name}': {e}")
            raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}") from e
    
    def embed_text(self, text: str) -> list:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as a list
            
        Raises:
            ValueError: If text is empty or None
            RuntimeError: If encoding fails
        """
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")
        
        if not text.strip():
            logger.warning("Empty text provided, returning zero vector")
            return [0.0] * self.dimension
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e
    
    def embed_batch(self, texts: list) -> list:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If texts is empty, None, or contains invalid entries
            RuntimeError: If encoding fails
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("texts must be a non-empty list")
        
        if len(texts) == 0:
            raise ValueError("texts list cannot be empty")
        
        # Validate all texts are strings
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a string, got {type(text)}")
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}") from e
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.dimension

