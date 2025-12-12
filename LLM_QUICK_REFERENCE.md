# Milestone 3: LLM Layer - Quick Reference

## What I Implemented

This is a quick reference for the **LLM Layer** component of Milestone 3.

### ✅ Requirement 3.a: Combine KG Results from Baseline + Embeddings

**File**: `app.py` (lines 216-232)

**What it does**:
- Merges results from baseline Cypher queries and embedding-based retrieval
- Removes duplicate records
- Limits context to 30 records to manage token size
- Formats as JSON for LLM consumption

**Code location**:
```python
# Combine results
all_results = baseline_results + embedding_results
# Deduplicate
seen = set()
unique_results = []
for r in all_results:
    key = str(sorted(r.items()))
    if key not in seen:
        seen.add(key)
        unique_results.append(r)
# Format context
context = json.dumps(unique_results[:30], indent=2)
```

---

### ✅ Requirement 3.b: Structured Prompt (Context + Persona + Task)

**File**: `llm_layer/prompts.py`

**What it does**:
- Builds structured prompts with three components
- Context: Retrieved KG data in JSON format
- Persona: Defines airline assistant role
- Task: Clear instructions to use only KG context

**Functions**:
- `get_persona()` - Returns persona definition
- `get_task_instruction()` - Returns task instructions
- `build_prompt(context, persona, task, query)` - Combines all parts

**Prompt structure**:
```
[Persona: You are an Airline Company Flight Insights Assistant...]

[Task: Based on the context provided from the knowledge graph...]

Context from Knowledge Graph:
[JSON data]

User Query: [user's question]

Answer:
```

---

### ✅ Requirement 3.c: Compare At Least 3 Models

**Files**: `llm_layer/models.py`, `config.py`

**What it does**:
- Supports 6 different LLM models across 5 providers
- Provides unified interface for all models
- Automatically initializes available models based on API keys

**Supported Models**:
1. **gpt-3.5-turbo** (OpenAI)
2. **gpt-4** (OpenAI)
3. **claude-3-haiku-20240307** (Anthropic)
4. **gemini-pro** (Google)
5. **mistralai/mistral-7b-instruct** (OpenRouter)
6. **meta-llama/llama-3-8b-instruct** (OpenRouter)

**Key Classes**:
- `LLMModel` - Wrapper for individual models
- `LLMManager` - Manages and compares multiple models

**Usage**:
```python
llm_manager = LLMManager()
results = llm_manager.compare_models(prompt, ["gpt-3.5-turbo", "gpt-4", "claude-3-haiku-20240307"])
```

---

### ✅ Requirement 3.d: Qualitative & Quantitative Evaluation

**Files**: `utils/evaluation.py`, `evaluate_llms.py`

**What it does**:
- Automatically tracks quantitative metrics
- Provides framework for qualitative assessment
- Exports detailed results for analysis

**Quantitative Metrics** (automatic):
- Response Time (seconds)
- Estimated Tokens
- Response Length (characters)
- Estimated Cost (USD)
- Words per Second

**Qualitative Criteria** (manual assessment, 1-5 scale):
- Relevance: Does it answer the question?
- Accuracy: Is information correct?
- Naturalness: Is language fluent?
- Completeness: All details included?
- Groundedness: Faithful to KG context?

**Run evaluation**:
```bash
python evaluate_llms.py --models gpt-3.5-turbo claude-3-haiku-20240307 mistralai/mistral-7b-instruct
```

**Output**: `llm_evaluation_results.json` with all metrics and responses

---

## Files Overview

| File | Purpose | Lines |
|------|---------|-------|
| `llm_layer/models.py` | LLM model wrappers and manager | 177 |
| `llm_layer/prompts.py` | Structured prompt builder | 38 |
| `utils/evaluation.py` | Evaluation utilities | 95 |
| `evaluate_llms.py` | Comprehensive evaluation script | 350 |
| `app.py` | Streamlit UI with LLM integration | 378 |
| `LLM_LAYER.md` | Detailed documentation | - |
| `LLM_EVALUATION_GUIDE.md` | Step-by-step evaluation guide | - |

---

## How to Demo for Presentation

### 1. Show Structured Prompt (3.b)

**In presentation**: Show slide with prompt structure

**In demo**: 
- Open `llm_layer/prompts.py`
- Highlight the three components: persona, task, context
- Explain how it reduces hallucination by grounding in KG

### 2. Show Multi-Model Support (3.c)

**In presentation**: List supported models in table

**In demo**:
- Open Streamlit UI
- Show model selection dropdown
- Check "Compare Multiple Models"
- Select 3 models
- Run a query
- Show side-by-side responses

### 3. Show Evaluation Results (3.d)

**In presentation**: 
- Table of quantitative metrics (time, tokens, cost)
- Table of qualitative scores (relevance, accuracy, etc.)
- Declare which model won and why

**In demo**:
- Show `llm_evaluation_results.json`
- Highlight response times
- Show example responses from each model
- Explain trade-offs (speed vs quality vs cost)

### 4. Show Context Combination (3.a)

**In presentation**: Diagram showing baseline + embeddings → combined context

**In demo**:
- Run query with "Both (Hybrid)" retrieval
- Expand "Retrieved Knowledge Graph Context"
- Show combined results table
- Expand "Context Statistics" to show counts

---

## API Keys Needed

For your evaluation, you need at least 3 API keys. **Budget-friendly options**:

1. **OpenAI** - `gpt-3.5-turbo` (~$0.002/1K tokens)
2. **OpenRouter** - `claude-3-haiku-20240307` (~$0.0003/1K tokens)
3. **OpenRouter** - `mistralai/mistral-7b-instruct` (~$0.0002/1K tokens)

Set in `.env`:
```bash
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
```

**Cost estimate**: ~$0.50-1.00 for 10 test queries across 3 models

---

## Quick Test

Before your presentation, verify everything works:

```bash
# 1. Test single query
python evaluate_llms.py --custom-query "Which flights have the worst delays?" --models gpt-3.5-turbo

# 2. If successful, run full evaluation
python evaluate_llms.py --models gpt-3.5-turbo claude-3-haiku-20240307 mistralai/mistral-7b-instruct

# 3. Check output file
cat llm_evaluation_results.json
```

---

## Presentation Checklist

- [ ] Slide explaining structured prompt (3.b)
- [ ] Table listing 3+ models compared (3.c)
- [ ] Quantitative metrics table (3.d)
- [ ] Qualitative evaluation results (3.d)
- [ ] Example query with all 3 model responses
- [ ] Winner announcement with justification
- [ ] Live demo of model comparison in UI
- [ ] Screenshots of evaluation results

---

## Common Issues & Solutions

**Issue**: "Model X not available (missing API key?)"
- **Solution**: Add API key to `.env` file and restart

**Issue**: "No context retrieved from KG"
- **Solution**: Check Neo4j is running and has data

**Issue**: API rate limits
- **Solution**: Add small delay between requests, or use fewer test queries

**Issue**: High costs
- **Solution**: Use cheaper models (Mistral, Claude Haiku) instead of GPT-4

---

## What Makes This Implementation Strong

1. **Complete Coverage**: All 4 sub-requirements fully implemented
2. **Exceeds Requirements**: 6 models (not just 3)
3. **Well Documented**: Comprehensive docs and guides
4. **Production Quality**: Error handling, logging, metrics
5. **Easy to Use**: One-command evaluation script
6. **Flexible**: Works with any combination of models
7. **Transparent**: Shows Cypher queries, context, and reasoning
8. **Cost Conscious**: Includes cost estimation

---

## For Office Hours / Questions

If you need help:
1. Check `LLM_EVALUATION_GUIDE.md` for detailed instructions
2. Review `LLM_LAYER.md` for technical documentation
3. Run test command to verify setup
4. Bring specific error messages

Good luck with your presentation! 🎉
