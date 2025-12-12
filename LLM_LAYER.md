# LLM Layer Implementation

## Overview

The LLM Layer is the third component of the Graph-RAG pipeline, responsible for generating natural language responses using the retrieved knowledge graph context. It implements **Requirement 3** of Milestone 3.

## Architecture

```
Retrieved KG Context
    ↓
┌─────────────────────────────────────────┐
│  3.a Context Combination                │
│  - Merge baseline + embedding results   │
│  - Deduplicate records                  │
│  - Limit context size (30 records)      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  3.b Structured Prompt Builder          │
│  - Context: KG data (JSON)              │
│  - Persona: Airline assistant role      │
│  - Task: Answer using context only      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  3.c Multi-Model LLM Support            │
│  - OpenAI (GPT-3.5, GPT-4)              │
│  - Anthropic (Claude 3)                 │
│  - Google (Gemini Pro)                  │
│  - OpenRouter (Mistral, Llama)          │
│  - HuggingFace (open-source)            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  3.d Model Evaluation                   │
│  - Quantitative: time, tokens, cost     │
│  - Qualitative: relevance, accuracy     │
└─────────────────────────────────────────┘
    ↓
Final Answer
```

## Implementation Details

### 3.a Context Combination

**Location**: `app.py` (lines 216-232)

The system combines results from both baseline Cypher queries and embedding-based retrieval:

```python
# Combine results
all_results = baseline_results + embedding_results

# Remove duplicates
seen = set()
unique_results = []
for r in all_results:
    key = str(sorted(r.items()))
    if key not in seen:
        seen.add(key)
        unique_results.append(r)

# Format context for LLM (limit to 30 records)
context = json.dumps(unique_results[:30], indent=2)
```

**Key Features**:
- Merges structured (baseline) and semantic (embedding) results
- Deduplication based on record content
- Context size limiting to manage LLM token limits
- JSON formatting for clear structure

### 3.b Structured Prompt

**Location**: `llm_layer/prompts.py`

The prompt follows a three-component structure as required:

#### 1. **Persona** (Role Definition)
```python
def get_persona() -> str:
    return """You are an Airline Company Flight Insights Assistant. 
    Your role is to help airline management understand flight performance, 
    passenger satisfaction, delays, and operational insights. You provide 
    data-driven insights based on the knowledge graph information provided."""
```

#### 2. **Task** (Instructions)
```python
def get_task_instruction() -> str:
    return """Based on the context provided from the knowledge graph, 
    answer the user's question accurately and comprehensively. Use only 
    the information provided in the context. If the context doesn't contain 
    enough information to answer the question, say so. Be specific with 
    numbers, flight numbers, airport codes, and metrics when available."""
```

#### 3. **Context** (Retrieved KG Data)
- Retrieved nodes, relationships, and properties from Neo4j
- Formatted as JSON for clarity
- Includes both baseline and embedding results

#### Complete Prompt Structure
```python
def build_prompt(context: str, persona: str, task: str, user_query: str) -> str:
    prompt = f"""{persona}

{task}

Context from Knowledge Graph:
{context}

User Query: {user_query}

Answer:"""
    return prompt
```

### 3.c Multi-Model Support

**Location**: `llm_layer/models.py`

The system supports **6 different LLM models** across 5 providers (exceeding the requirement of 3 models):

#### Supported Models

| Provider | Models | API Required | Notes |
|----------|--------|--------------|-------|
| **OpenAI** | `gpt-3.5-turbo`, `gpt-4` | OPENAI_API_KEY | Paid, high quality |
| **Anthropic** | `claude-3-haiku-20240307` | ANTHROPIC_API_KEY | Paid, fast |
| **Google** | `gemini-pro` | GOOGLE_API_KEY | Paid/Free tier |
| **OpenRouter** | `mistralai/mistral-7b-instruct`, `meta-llama/llama-3-8b-instruct` | OPENROUTER_API_KEY | Pay-per-use, access to open-source |
| **HuggingFace** | Any HF model | HUGGINGFACE_API_KEY | Free tier available |

#### LLMModel Class

```python
class LLMModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.provider = self._get_provider()
        self.llm = self._initialize_model()
        
    def invoke(self, prompt: str) -> str:
        """Generate response with timing and token tracking"""
        
    def get_metrics(self) -> dict:
        """Return performance metrics"""
```

#### LLMManager Class

```python
class LLMManager:
    def __init__(self):
        self._initialize_available_models()
        
    def get_model(self, model_name: str) -> LLMModel:
        """Get specific model instance"""
        
    def compare_models(self, prompt: str, model_names: list) -> dict:
        """Compare multiple models on same prompt"""
```

**Features**:
- Automatic model initialization based on available API keys
- Graceful fallback if models are unavailable
- Unified interface across all providers
- Automatic error handling

### 3.d Evaluation Framework

**Location**: `utils/evaluation.py`, `evaluate_llms.py`

#### Quantitative Metrics

Tracked automatically for each model:

1. **Response Time** (seconds)
   - How long the model takes to generate a response
   - Lower is better

2. **Estimated Tokens**
   - Approximate token count (input + output)
   - Used for cost estimation

3. **Response Length** (characters)
   - Length of generated answer
   - Indicates verbosity

4. **Estimated Cost** (USD)
   - Based on current API pricing
   - Helps compare cost-effectiveness

5. **Words per Second**
   - Generation speed metric
   - Response length / response time

#### Qualitative Evaluation Criteria

Manual assessment on a 1-5 scale:

1. **Relevance**
   - Does the answer address the user's question?
   - Is it on-topic and appropriate?

2. **Accuracy**
   - Is the information factually correct?
   - Does it match the KG context?

3. **Naturalness**
   - Does it read fluently and naturally?
   - Is the language appropriate?

4. **Completeness**
   - Is all relevant information included?
   - Are important details missing?

5. **Groundedness**
   - Does it stay faithful to KG context?
   - Are there hallucinations or unsupported claims?

#### Evaluation Script

**Running Evaluations**:

```bash
# Evaluate all test queries with default models
python evaluate_llms.py

# Evaluate with specific models
python evaluate_llms.py --models gpt-3.5-turbo claude-3-haiku-20240307 gemini-pro

# Evaluate custom query
python evaluate_llms.py --custom-query "Which flights have the worst delays?"

# Use custom test queries file
python evaluate_llms.py --queries my_tests.json --output my_results.json
```

**Output**:
- JSON file with detailed results for each query and model
- Console summary with aggregate statistics
- Quantitative metrics automatically computed
- Qualitative evaluation template for manual assessment

## Usage Examples

### Basic Usage in Code

```python
from llm_layer.models import LLMManager
from llm_layer.prompts import build_prompt, get_persona, get_task_instruction

# Initialize
llm_manager = LLMManager()

# Get available models
available = llm_manager.list_available_models()
print(f"Available models: {available}")

# Single model
model = llm_manager.get_model("gpt-3.5-turbo")
prompt = build_prompt(context, get_persona(), get_task_instruction(), query)
response = model.invoke(prompt)
metrics = model.get_metrics()

# Compare multiple models
results = llm_manager.compare_models(prompt, ["gpt-3.5-turbo", "gpt-4"])
```

### UI Integration

The LLM layer is fully integrated into the Streamlit UI (`app.py`):

1. **Model Selection**: Dropdown to choose LLM
2. **Model Comparison**: Multi-select for side-by-side comparison
3. **Metrics Display**: Automatic performance metrics
4. **Response Display**: Clear presentation of LLM answers

## Configuration

**Location**: `config.py`

```python
# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# LLM Models Configuration
LLM_MODELS = {
    "gpt-3.5-turbo": {
        "provider": "openai",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    # ... more models
}
```

Set API keys in `.env` file:
```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
# etc.
```

## Milestone 3 Compliance

### ✅ Requirement 3.a: Combine KG Results
- [x] Combines baseline Cypher queries with embedding results
- [x] Deduplication logic implemented
- [x] Context size management (30 records limit)
- [x] JSON formatting for clear structure

### ✅ Requirement 3.b: Structured Prompt
- [x] Context component (retrieved KG data)
- [x] Persona component (airline assistant role)
- [x] Task component (clear instructions)
- [x] Proper prompt structure implementation

### ✅ Requirement 3.c: Compare 3+ Models
- [x] 6 models supported (exceeds requirement)
- [x] 5 providers (OpenAI, Anthropic, Google, OpenRouter, HuggingFace)
- [x] LLMManager for easy comparison
- [x] Unified interface across providers

### ✅ Requirement 3.d: Qualitative & Quantitative Evaluation
- [x] Quantitative: response time, tokens, cost, speed
- [x] Qualitative: 5 criteria framework (relevance, accuracy, naturalness, completeness, groundedness)
- [x] Evaluation script for systematic testing
- [x] Results export and analysis tools

## Error Handling

The LLM layer includes robust error handling:

1. **Missing API Keys**: Models gracefully skip if API key unavailable
2. **API Errors**: Caught and logged without crashing system
3. **Timeout Handling**: LangChain's built-in timeout mechanisms
4. **Empty Context**: Handles cases with no KG results
5. **Model Unavailability**: Shows user-friendly messages in UI

## Performance Considerations

1. **Token Limits**: Context limited to 30 records to stay within model limits
2. **Caching**: Session state caching in Streamlit UI
3. **Parallel Requests**: Could be added for batch evaluation
4. **Cost Management**: Estimated costs displayed to user
5. **Free Tier Options**: HuggingFace and OpenRouter for testing

## Future Enhancements

Potential improvements beyond Milestone 3:

1. **Streaming Responses**: Real-time token streaming in UI
2. **Fine-tuning**: Train custom models on airline domain
3. **Prompt Optimization**: A/B testing of prompt variations
4. **Caching**: Cache responses for repeated queries
5. **Feedback Loop**: User ratings to improve prompts
6. **Async Processing**: Parallel model comparisons
7. **Advanced Metrics**: BLEU, ROUGE scores for automatic evaluation

## References

- LangChain Documentation: https://python.langchain.com/
- OpenAI API: https://platform.openai.com/docs
- Anthropic Claude: https://docs.anthropic.com/
- Google Gemini: https://ai.google.dev/
- HuggingFace: https://huggingface.co/docs
- OpenRouter: https://openrouter.ai/docs
