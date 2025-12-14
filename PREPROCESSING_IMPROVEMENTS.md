# Preprocessing Improvements

This document outlines specific, actionable improvements to address the errors identified in the preprocessing error analysis.

## Priority 1: High Impact Improvements

### 1. Expand Intent Patterns (Addresses 95% Unrecognized Intent Rate)

**Current Issue**: Many common airline queries fall back to `general_question`.

**Improvements**:

#### Add New Intent Categories:
```python
"booking": [
    r"book.*flight",
    r"reserve.*flight",
    r"purchase.*ticket",
    r"buy.*ticket"
],
"cancellation": [
    r"cancel.*flight",
    r"cancel.*reservation",
    r"cancel.*booking",
    r"refund"
],
"check_in": [
    r"check.*in",
    r"online.*check.*in",
    r"web.*check.*in"
],
"flight_status": [
    r"flight.*status",
    r"where.*flight",
    r"is.*flight.*on.*time",
    r"flight.*delayed"
],
"seat_selection": [
    r"select.*seat",
    r"choose.*seat",
    r"seat.*selection",
    r"preferred.*seat"
],
"baggage": [
    r"baggage",
    r"luggage",
    r"carry.*on",
    r"checked.*bag",
    r"baggage.*allowance"
],
"loyalty": [
    r"frequent.*flyer",
    r"loyalty.*program",
    r"miles",
    r"points",
    r"status"
]
```

**Expected Impact**: Reduce unrecognized intent rate from ~95% to ~30-40%

### 2. Expand Airport Code Whitelist (Addresses 10.5% False Negatives)

**Current Issue**: Many valid international airport codes are not recognized.

**Improvements**:

Add comprehensive IATA airport code database:
```python
# Add to valid_airport_codes set:
# Major international airports (missing codes)
"BKK", "KUL", "SIN", "ICN", "CAN", "CPH", "OSL", "ARN", "HEL",
"VIE", "ZUR", "MAD", "BCN", "FCO", "MUC", "DUB", "MAN", "YYZ",
"YVR", "YUL", "SYD", "MEL", "AKL", "JNB", "CAI", "IST", "DXB",
"AUH", "DOH", "KWI", "BAH", "RUH", "JED", "DMM", "RIY", "BOM",
"DEL", "CCU", "MAA", "BLR", "HYD", "PNQ", "COK", "CCJ", "TRV",
"BKK", "DMK", "CNX", "HKT", "KBV", "USM", "HDY", "NST", "UBP",
"KUL", "PEN", "LGK", "BKI", "KCH", "MYY", "TWU", "LBU", "SDK",
"SIN", "BWN", "KUL", "CGK", "DPS", "SUB", "UPG", "MDC", "PLW",
"ICN", "GMP", "PUS", "CJU", "TAE", "KUV", "WJU", "YNY", "RSU",
"PEK", "PVG", "CAN", "SZX", "CTU", "XIY", "KMG", "URC", "XMN",
"TAO", "TSN", "DLC", "NGB", "HGH", "NKG", "WUH", "CSX", "CGO",
"TYN", "SJW", "HRB", "CGQ", "DNH", "YNT", "JHG", "KWE", "KWL"
```

**Alternative Approach**: Use a comprehensive airport code database file (JSON/CSV) that can be loaded dynamically.

**Expected Impact**: Reduce false negative rate from ~10.5% to ~2-3%

### 3. Support More Date Formats (Addresses 50% Unrecognized Date Formats)

**Current Issue**: Many common date formats are not recognized.

**Improvements**:

Add additional date patterns:
```python
self.date_patterns = [
    r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD (existing)
    r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY (existing)
    r'\b\d{1,2}/\d{1,2}/\d{2}\b',  # MM/DD/YY or DD/MM/YY (NEW)
    r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # DD-MM-YYYY or MM-DD-YYYY (NEW)
    r'\b\d{1,2}\.\d{1,2}\.\d{4}\b',  # DD.MM.YYYY or MM.DD.YYYY (NEW)
    r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',  # Full month (existing)
    r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',  # Abbreviated month (NEW)
    r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',  # Ordinal date (NEW)
]
```

**Add Relative Date Support**:
```python
def extract_relative_dates(self, query: str) -> list:
    """Extract relative dates like 'tomorrow', 'next Monday', etc."""
    relative_patterns = {
        r'\btomorrow\b': 1,  # days from today
        r'\btoday\b': 0,
        r'\byesterday\b': -1,
        r'\bnext\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b': None,  # Calculate next weekday
        r'\bthis\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b': None,
        r'\bnext\s+week\b': 7,
        r'\bthis\s+week\b': 0,
        r'\bnext\s+month\b': None,  # Calculate next month
    }
    # Implementation would convert relative dates to absolute dates
    # using datetime calculations
```

**Expected Impact**: Reduce unrecognized date format rate from ~50% to ~10-15%

## Priority 2: Medium Impact Improvements

### 4. Improve Ambiguous Query Handling (Addresses 10% Misclassification)

**Current Issue**: Queries like "Show me flights" could be `flight_search` or `general_question`.

**Improvements**:

#### Add Context-Aware Scoring:
```python
def classify(self, query: str) -> str:
    query_lower = query.lower()
    
    # Score each intent
    intent_scores = {}
    for intent, patterns in self.intent_patterns.items():
        score = sum(1 for pattern in patterns if re.search(pattern, query_lower, re.IGNORECASE))
        if score > 0:
            intent_scores[intent] = score
    
    # Context-aware disambiguation
    if len(intent_scores) > 1:
        # If multiple intents match, use heuristics
        if "flight" in query_lower and "flight_search" in intent_scores:
            # Prefer flight_search over general_question for flight-related queries
            if "general_question" in intent_scores:
                intent_scores["flight_search"] += 2  # Boost flight_search
        # Similar heuristics for other ambiguous cases
    
    # Return the intent with highest score
    if intent_scores:
        return max(intent_scores, key=intent_scores.get)
    return "general_question"
```

#### Add Query Length Heuristics:
- Short queries (1-2 words) with "flights" → `flight_search`
- Queries with action verbs (find, show, get) + entity → specific intent
- Queries with question words only → `general_question`

**Expected Impact**: Reduce ambiguous query misclassification from ~10% to ~3-5%

### 5. Multi-Intent Detection

**Current Issue**: Queries with multiple intents only return one intent.

**Improvements**:

#### Return Multiple Intents:
```python
def classify_multiple(self, query: str, max_intents: int = 2) -> list:
    """Classify query and return top N intents."""
    query_lower = query.lower()
    
    intent_scores = {}
    for intent, patterns in self.intent_patterns.items():
        score = sum(1 for pattern in patterns if re.search(pattern, query_lower, re.IGNORECASE))
        if score > 0:
            intent_scores[intent] = score
    
    # Return top N intents sorted by score
    sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
    return [intent for intent, score in sorted_intents[:max_intents]]
```

#### Update Retrieval Layer:
- Modify `BaselineRetriever` to handle multiple intents
- Execute queries for each intent and combine results
- Use intent priority/weighting for result ranking

**Expected Impact**: Enable handling of complex queries like "Compare delays and satisfaction"

### 6. Improve Entity Extraction for Missing Entities

**Current Issue**: Some queries don't extract expected entities.

**Improvements**:

#### Add Implicit Entity Inference:
```python
def extract_entities_with_inference(self, query: str, intent: str) -> dict:
    """Extract entities with context-aware inference."""
    entities = self.extract_entities(query)
    
    # If intent is flight_search but no airports found, check for city names
    if intent == "flight_search" and "AIRPORT" not in entities:
        # Try to extract city names and map to airports
        city_airports = self._extract_city_names(query)
        if city_airports:
            entities["AIRPORT"] = city_airports
    
    # If intent is delay_analysis but no flights found, infer from context
    if intent == "delay_analysis" and "FLIGHT" not in entities:
        # Could infer from route or date context
    
    return entities
```

#### Add City Name to Airport Code Mapping:
```python
self.city_to_airport = {
    "new york": ["JFK", "LGA", "EWR"],
    "los angeles": ["LAX"],
    "chicago": ["ORD", "MDW"],
    "london": ["LHR", "LGW", "STN"],
    "paris": ["CDG", "ORY"],
    # ... more mappings
}
```

**Expected Impact**: Reduce missing entity rate for context-aware queries

## Priority 3: Low Impact / Nice-to-Have Improvements

### 7. Add Fuzzy Matching for Airport Codes

**Improvements**:
- Handle typos in airport codes (e.g., "JFK" vs "JKF")
- Use Levenshtein distance for similarity matching
- Suggest corrections for invalid codes

### 8. Add Temporal Expression Parsing

**Improvements**:
- Better handling of relative dates ("next week", "in 3 days")
- Support for date ranges ("between March 1 and March 15")
- Holiday and event-based dates ("Christmas", "New Year")

### 9. Add Confidence Scores

**Improvements**:
- Return confidence scores for intent classification
- Return confidence scores for entity extraction
- Use low-confidence results to trigger clarification questions

### 10. Add Query Normalization

**Improvements**:
- Normalize abbreviations ("NYC" → "New York")
- Handle common misspellings
- Expand contractions ("don't" → "do not")
- Remove filler words

## Implementation Priority

1. **Immediate (Week 1)**:
   - Expand intent patterns (Priority 1, #1)
   - Expand airport code whitelist (Priority 1, #2)
   - Support more date formats (Priority 1, #3)

2. **Short-term (Week 2-3)**:
   - Improve ambiguous query handling (Priority 2, #4)
   - Add implicit entity inference (Priority 2, #6)

3. **Medium-term (Month 1-2)**:
   - Multi-intent detection (Priority 2, #5)
   - Add confidence scores (Priority 3, #9)

4. **Long-term (Month 2+)**:
   - Fuzzy matching (Priority 3, #7)
   - Temporal expression parsing (Priority 3, #8)
   - Query normalization (Priority 3, #10)

## Expected Overall Impact

After implementing Priority 1 improvements:
- **Unrecognized Intent Rate**: 95% → 30-40% (55-65% improvement)
- **False Negative Rate**: 10.5% → 2-3% (70-80% improvement)
- **Date Format Recognition**: 50% → 10-15% (70-80% improvement)

After implementing Priority 2 improvements:
- **Ambiguous Query Misclassification**: 10% → 3-5% (50-70% improvement)
- **Multi-intent Support**: 0% → 100% (new capability)

## Testing Recommendations

1. Create comprehensive test suite with all error cases
2. Add regression tests for each improvement
3. Measure improvement metrics after each change
4. Use A/B testing to validate improvements in production

