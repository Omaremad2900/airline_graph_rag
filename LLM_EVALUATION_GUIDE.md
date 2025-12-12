# LLM Evaluation Quick Guide

This guide helps you run the LLM evaluation experiments required for Milestone 3.

## Prerequisites

1. **Neo4j Running**: Database with airline knowledge graph loaded
2. **API Keys**: At least 3 LLM API keys configured in `.env`
3. **Embeddings**: FAISS indices initialized (optional but recommended)

## Step 1: Configure API Keys

Edit your `.env` file with at least 3 API keys:

```bash
# Recommended: Use free/low-cost options for testing
OPENAI_API_KEY=sk-...                    # GPT-3.5 is affordable
HUGGINGFACE_API_KEY=hf_...              # Free tier available
OPENROUTER_API_KEY=sk-or-...            # Pay-per-use, multiple models

# Optional: Additional models
ANTHROPIC_API_KEY=sk-ant-...            # Claude (paid)
GOOGLE_API_KEY=...                       # Gemini (free tier)
```

### Getting API Keys

- **OpenAI**: https://platform.openai.com/api-keys (requires payment)
- **HuggingFace**: https://huggingface.co/settings/tokens (free tier)
- **OpenRouter**: https://openrouter.ai/keys (pay-per-use, cheap)
- **Anthropic**: https://console.anthropic.com/ (requires payment)
- **Google**: https://makersuite.google.com/app/apikey (free tier)

**Budget-Friendly Recommendation**:
```bash
# Use these 3 for cost-effective evaluation:
1. gpt-3.5-turbo (OpenAI) - ~$0.002 per 1K tokens
2. claude-3-haiku-20240307 (via OpenRouter) - ~$0.0003 per 1K tokens
3. mistralai/mistral-7b-instruct (via OpenRouter) - ~$0.0002 per 1K tokens
```

## Step 2: Prepare Test Queries

The system includes default test queries in `tests/test_queries.json`. You can use these or create your own.

**Example test queries**:
```json
[
    {
        "query": "Which flights have the worst delays?",
        "intent": "delay_analysis"
    },
    {
        "query": "Show me flights from JFK to LAX",
        "intent": "flight_search"
    },
    {
        "query": "What are the routes with low passenger satisfaction?",
        "intent": "passenger_satisfaction"
    }
]
```

## Step 3: Run Quick Test

Test with a single query to verify everything works:

```bash
python evaluate_llms.py --custom-query "Which flights have the worst delays?" --models gpt-3.5-turbo
```

**Expected Output**:
```
================================================================================
Query: Which flights have the worst delays?
================================================================================
Retrieving context from KG...
Retrieved: 10 baseline, 0 embedding, 10 total unique

Comparing 1 models...
  - Testing gpt-3.5-turbo...
    ✓ Response time: 2.34s, Est. tokens: 450
```

## Step 4: Run Full Evaluation

Evaluate all test queries with your chosen models:

```bash
# Using 3 models (adjust based on your API keys)
python evaluate_llms.py --models gpt-3.5-turbo claude-3-haiku-20240307 mistralai/mistral-7b-instruct
```

**This will**:
1. Load all queries from `tests/test_queries.json`
2. Retrieve KG context for each query
3. Generate responses from all 3 models
4. Save results to `llm_evaluation_results.json`
5. Print summary statistics

**Output Files**:
- `llm_evaluation_results.json` - Detailed results with all responses and metrics

## Step 5: Review Results

### Quantitative Metrics (Automatic)

Open `llm_evaluation_results.json` and check:
- `response_time` - How fast each model responded
- `estimated_tokens` - Token usage (affects cost)
- `response_length` - Length of generated answers
- `estimated_cost` - Approximate API cost

**Summary is printed to console**:
```
Model                          Success    Failed  Avg Time   Avg Tokens   Avg Length
---------------------------------------------------------------------------------------------
gpt-3.5-turbo                       10         0      2.34s          450          850
claude-3-haiku-20240307            10         0      1.89s          420          780
mistralai/mistral-7b-instruct      10         0      3.12s          480          920
```

### Qualitative Evaluation (Manual)

For each model's response, manually assess on a 1-5 scale:

1. **Relevance**: Does it answer the question? (1=not relevant, 5=perfectly relevant)
2. **Accuracy**: Is information correct per KG? (1=incorrect, 5=fully accurate)
3. **Naturalness**: Is language fluent? (1=awkward, 5=natural)
4. **Completeness**: All details included? (1=missing info, 5=complete)
5. **Groundedness**: Stays true to KG context? (1=hallucinations, 5=fully grounded)

**Create a spreadsheet** or document like:

| Query | Model | Relevance | Accuracy | Naturalness | Completeness | Groundedness | Notes |
|-------|-------|-----------|----------|-------------|--------------|--------------|-------|
| "Which flights..." | GPT-3.5 | 5 | 4 | 5 | 4 | 5 | Good, mentioned specific flights |
| "Which flights..." | Claude | 5 | 5 | 4 | 5 | 5 | Very accurate, slight awkward phrasing |
| "Which flights..." | Mistral | 4 | 4 | 3 | 3 | 4 | Relevant but less complete |

## Step 6: Generate Comparison Report

Based on your evaluation, create a summary for your presentation:

### Example Summary

**Quantitative Comparison**:
```
Model Performance Summary (10 test queries):

GPT-3.5-Turbo:
  - Avg Response Time: 2.34s
  - Avg Tokens: 450
  - Avg Cost: $0.001 per query
  - Success Rate: 100%

Claude 3 Haiku:
  - Avg Response Time: 1.89s (19% faster)
  - Avg Tokens: 420 (7% fewer)
  - Avg Cost: $0.0004 per query (60% cheaper)
  - Success Rate: 100%

Mistral 7B:
  - Avg Response Time: 3.12s (33% slower)
  - Avg Tokens: 480 (7% more)
  - Avg Cost: $0.0002 per query (80% cheaper)
  - Success Rate: 100%
```

**Qualitative Comparison**:
```
Average Scores (out of 5):

                 Relevance  Accuracy  Naturalness  Completeness  Groundedness
GPT-3.5-Turbo       4.8       4.5         4.9          4.6           4.7
Claude Haiku        4.7       4.8         4.6          4.8           4.9
Mistral 7B          4.3       4.1         3.8          3.9           4.2

Best for: Speed → Claude Haiku
Best for: Cost → Mistral 7B
Best for: Accuracy → Claude Haiku
Best for: Naturalness → GPT-3.5-Turbo
Overall Winner: Claude 3 Haiku (best balance of speed, cost, and quality)
```

## Advanced Options

### Custom Models

```bash
# Test with different model combinations
python evaluate_llms.py --models gpt-4 gemini-pro meta-llama/llama-3-8b-instruct
```

### Custom Test Queries

Create your own `my_tests.json`:
```json
[
    {"query": "Your custom query 1"},
    {"query": "Your custom query 2"}
]
```

Run with:
```bash
python evaluate_llms.py --queries my_tests.json --output my_results.json
```

### Batch Processing

For large evaluations, run in batches to avoid API rate limits:
```bash
# First 5 queries
python evaluate_llms.py --queries tests/test_queries.json --output batch1.json

# Wait a few minutes, then continue...
```

## Troubleshooting

### "Model X not available (missing API key?)"
- Check that API key is set in `.env`
- Verify key format (should start with correct prefix: `sk-`, `hf_`, etc.)
- Restart Python if you just added the key

### "Failed to connect to Neo4j"
- Ensure Neo4j is running
- Check connection settings in `.env`
- Verify database contains data

### "No context retrieved from KG"
- Check that embeddings are initialized (if using embedding retrieval)
- Verify test queries match your data
- Try simpler queries first

### API Rate Limits
- Add delays between requests
- Use fewer queries initially
- Consider using OpenRouter for better rate limits

### High Costs
- Start with free models (HuggingFace)
- Use cheaper models (Mistral, Claude Haiku)
- Limit test queries to 5-10 initially
- Check pricing: https://openrouter.ai/docs#models

## For Your Presentation

Include in your slides:

1. **Models Compared**: List the 3+ models you tested
2. **Quantitative Table**: Response time, tokens, cost comparison
3. **Qualitative Table**: Manual evaluation scores
4. **Example Responses**: Show 1-2 example queries with each model's response
5. **Winner**: Which model is best for this use case and why
6. **Trade-offs**: Speed vs quality vs cost analysis

## Tips for Success

1. **Test Early**: Run evaluations well before the deadline
2. **Start Small**: Test with 2-3 queries first, then scale up
3. **Document Everything**: Save all JSON outputs
4. **Manual Review**: Don't skip qualitative evaluation
5. **Budget Wisely**: Use free/cheap models for initial testing
6. **Screenshots**: Capture UI showing model comparison
7. **Error Cases**: Document when models fail or hallucinate

## Checklist for Milestone 3 Submission

- [ ] At least 3 LLM models configured and tested
- [ ] Quantitative metrics collected (time, tokens, cost)
- [ ] Qualitative evaluation completed (relevance, accuracy, etc.)
- [ ] Results saved in JSON file
- [ ] Comparison summary prepared
- [ ] Example responses documented
- [ ] Screenshots of UI model comparison
- [ ] Error analysis included (if any failures occurred)
- [ ] Best model identified with justification

## Need Help?

- Check `LLM_LAYER.md` for detailed documentation
- Review `evaluate_llms.py` code for customization
- Ask during office hours
- Test with simple queries first to verify setup

Good luck with your evaluation! 🚀
