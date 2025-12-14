"""LLM model wrappers for different providers."""
import time
from typing import Optional
import config
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub
# from langchain_groq import ChatGroq


class LLMModel:
    """Base class for LLM models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_config = config.LLM_MODELS.get(model_name, {})
        self.provider = self.model_config.get("provider", "openai")
        self.llm = self._initialize_model()
        self.token_count = 0
        self.response_time = 0.0
    
    def _initialize_model(self):
        """Initialize the LLM model based on provider."""
        if self.provider == "openai":
            if not config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found")
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.model_config.get("temperature", 0.7),
                max_tokens=self.model_config.get("max_tokens", 1000),
                api_key=config.OPENAI_API_KEY
            )
        
        elif self.provider == "anthropic":
            if not config.ANTHROPIC_API_KEY:
                raise ValueError("Anthropic API key not found")
            return ChatAnthropic(
                model=self.model_name,
                temperature=self.model_config.get("temperature", 0.7),
                max_tokens=self.model_config.get("max_tokens", 1000),
                anthropic_api_key=config.ANTHROPIC_API_KEY
            )
        
        elif self.provider == "google":
            if not config.GOOGLE_API_KEY:
                raise ValueError("Google API key not found")
            # Handle model name - remove 'models/' prefix if present
            model_name = self.model_name
            if model_name.startswith("models/"):
                model_name = model_name.replace("models/", "")
            
            # Initialize ChatGoogleGenerativeAI
            # Note: gemini-pro is deprecated, use gemini-1.5-flash or gemini-1.5-pro
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.model_config.get("temperature", 0.7),
                max_output_tokens=self.model_config.get("max_tokens", 1000),
                google_api_key=config.GOOGLE_API_KEY
            )
        
        elif self.provider == "openrouter":
            if not config.OPENROUTER_API_KEY:
                raise ValueError("OpenRouter API key not found")
            # Validate API key format (should start with sk-or-)
            api_key = config.OPENROUTER_API_KEY.strip()
            if not api_key.startswith("sk-or-"):
                raise ValueError(
                    f"OpenRouter API key should start with 'sk-or-'. "
                    f"Your key starts with '{api_key[:7]}...'. "
                    f"Please check your .env file and get a valid key from https://openrouter.ai/keys"
                )
            # Use OpenAI-compatible interface for OpenRouter
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.model_config.get("temperature", 0.7),
                max_tokens=self.model_config.get("max_tokens", 1000),
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        
        elif self.provider == "huggingface":
            if not config.HUGGINGFACE_API_KEY:
                raise ValueError("HuggingFace API key not found")
            return HuggingFaceHub(
                repo_id=self.model_name,
                model_kwargs={
                    "temperature": self.model_config.get("temperature", 0.7),
                    "max_new_tokens": self.model_config.get("max_tokens", 1000)
                },
                huggingfacehub_api_token=config.HUGGINGFACE_API_KEY
            )
        
        elif self.provider == "groq":
            if not config.GROQ_API_KEY:
                raise ValueError("Groq API key not found")
            return ChatGroq(
                model=self.model_name,  # ChatGroq uses 'model' parameter
                temperature=self.model_config.get("temperature", 0.7),
                max_tokens=self.model_config.get("max_tokens", 1000),
                groq_api_key=config.GROQ_API_KEY
            )
        
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def invoke(self, prompt: str) -> str:
        """
        Invoke the LLM with a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            LLM response text
        """
        start_time = time.time()
        try:
            response = self.llm.invoke(prompt)
            
            # Extract text from response
            if hasattr(response, 'content'):
                text = response.content
            elif isinstance(response, str):
                text = response
            else:
                text = str(response)
            
            self.response_time = time.time() - start_time
            
            # Estimate token count (rough approximation)
            self.token_count = len(prompt.split()) + len(text.split())
            
            return text
        
        except Exception as e:
            print(f"Error invoking {self.model_name}: {e}")
            return f"Error: {str(e)}"
    
    def get_metrics(self) -> dict:
        """Get performance metrics for the last invocation."""
        return {
            "model": self.model_name,
            "provider": self.provider,
            "response_time": self.response_time,
            "estimated_tokens": self.token_count
        }


class LLMManager:
    """Manages multiple LLM models for comparison."""
    
    def __init__(self):
        self.models = {}
        self.available_models = []
        self._initialize_available_models()
    
    def _initialize_available_models(self):
        """Initialize available models based on API keys."""
        for model_name in config.LLM_MODELS.keys():
            try:
                model = LLMModel(model_name)
                self.models[model_name] = model
                self.available_models.append(model_name)
            except Exception as e:
                print(f"Could not initialize {model_name}: {e}")
    
    def get_model(self, model_name: str) -> Optional[LLMModel]:
        """Get a specific model by name."""
        return self.models.get(model_name)
    
    def list_available_models(self) -> list:
        """List all available models."""
        return self.available_models
    
    def compare_models(self, prompt: str, model_names: list = None) -> dict:
        """
        Compare multiple models on the same prompt.
        
        Args:
            prompt: Input prompt
            model_names: List of model names to compare (default: all available)
            
        Returns:
            Dictionary of model responses and metrics
        """
        if model_names is None:
            model_names = self.available_models
        
        results = {}
        for model_name in model_names:
            if model_name in self.models:
                model = self.models[model_name]
                response = model.invoke(prompt)
                metrics = model.get_metrics()
                results[model_name] = {
                    "response": response,
                    "metrics": metrics
                }
        
        return results

