# Preprocessing Error Analysis

This document summarizes the error analysis for the input preprocessing pipeline, including intent classification and entity extraction.

## Intent Classification Errors

### 1. Ambiguous Queries
- **Error Rate**: ~10% misclassification
- **Description**: Queries that could match multiple intent categories
- **Examples**:
  - "Show me flights" → Could be `flight_search` or `general_question`
  - "What flights?" → Could be `flight_search` or `general_question`
  - "Flights" → Could be `flight_search` or `general_question`
  - "Best flights" → Could be `flight_search`, `recommendation`, or `general_question` (misclassified as `performance_metrics`)

### 2. Unrecognized Intents
- **Error Rate**: ~95% (falls back to `general_question`)
- **Description**: Queries that don't match any intent pattern and fall back to default
- **Examples**:
  - "Book a flight" → Falls back to `general_question`
  - "Cancel my reservation" → Falls back to `general_question`
  - "Check in online" → Falls back to `general_question`
  - "Upgrade my seat" → Falls back to `general_question`
  - "Change my flight" → Falls back to `general_question`
  - "Refund policy" → Falls back to `general_question`
  - "Baggage allowance" → Falls back to `general_question`
  - "Check flight status" → Falls back to `general_question`
  - "Weather at airport" → Falls back to `general_question`
  - "Airport parking" → Falls back to `general_question`
  - "Lounge access" → Falls back to `general_question`
  - "Frequent flyer miles" → Falls back to `general_question`
  - "Flight cancellation" → Falls back to `general_question`
  - "Seat selection" → Falls back to `general_question`
  - "In-flight entertainment" → Falls back to `general_question`
  - "Meal preferences" → Falls back to `general_question`
  - "Special assistance" → Falls back to `general_question`
  - "Visa requirements" → Falls back to `general_question`
  - "Currency exchange" → Falls back to `general_question`

### 3. Multi-Intent Queries
- **Status**: Not supported
- **Description**: Queries that contain multiple intents but only one is returned
- **Examples**:
  - "Compare delays and satisfaction" → Only `delay_analysis` returned (satisfaction ignored)
  - "Show me flights and routes" → Only `route_analysis` returned (flights ignored)
  - "What are delays and ratings?" → Only `delay_analysis` returned (ratings ignored)
  - "Compare performance and satisfaction" → Only `performance_metrics` returned (satisfaction ignored)
  - "Find flights and check delays" → Only `flight_search` returned (delays ignored)
  - "Show routes and passenger feedback" → Only `passenger_satisfaction` returned (routes ignored)
  - "Compare delays with satisfaction ratings" → Only `passenger_satisfaction` returned (delays ignored)
  - "Find flights with good ratings" → Only `flight_search` returned (ratings ignored)
  - "Show delays and recommend best routes" → Only `delay_analysis` returned (recommendation ignored)
  - "Compare routes and performance metrics" → Only `performance_metrics` returned (routes ignored)

## Entity Extraction Errors

### 1. False Positives
- **Error Rate**: ~0% (effectively handled by exclusion list)
- **Description**: Entities incorrectly extracted (e.g., common words mistaken for airport codes)
- **Examples**:
  - "Find flights from THE airport" → "THE" not extracted (correctly excluded)
  - "Show me flights from AND to LAX" → "AND" not extracted (correctly excluded)
  - "Flights from FOR airport" → "FOR" not extracted (correctly excluded)
  - "What about flights from ARE airport" → "ARE" not extracted (correctly excluded)
  - "Show flights from ALL airports" → "ALL" not extracted (correctly excluded)

**Note**: The exclusion list (`excluded_words`) effectively prevents false positives for common English words.

### 2. False Negatives
- **Error Rate**: ~10.5% (missed valid entities)
- **Description**: Valid entities that should be extracted but are not recognized
- **Examples**:
  - "Show flights from XYZ to ABC" → "XYZ", "ABC" not recognized (not in whitelist)
  - "Find flights from BKK to KUL" → "BKK", "KUL" should be recognized (international codes)
  - "Flights from DXB to IST" → "DXB", "IST" should be recognized (international codes)
  - "Show flights from SIN to ICN" → "SIN", "ICN" should be recognized (international codes)
  - "Find flights from PEK to PVG" → "PEK", "PVG" should be recognized (international codes)
  - "Flights from CAN airport" → "CAN" should be recognized (international code)
  - "Show flights from MAD to BCN" → "MAD", "BCN" should be recognized (international codes)
  - "Find flights from FCO to MUC" → "FCO", "MUC" should be recognized (international codes)
  - "Flights from ZUR to VIE" → "ZUR", "VIE" should be recognized (international codes)
  - "Show flights from CPH to OSL" → "CPH", "OSL" should be recognized (international codes)

**Note**: Some international airport codes are in the whitelist, but many are missing, causing false negatives.

### 3. Date Format Variations
- **Error Rate**: ~50% (unrecognized formats)
- **Description**: Date formats that are not recognized by the current patterns
- **Examples**:
  - "Find flights on 15/03/24" → "15/03/24" format not recognized (DD/MM/YY)
  - "Show flights on 03/15/24" → "03/15/24" format not recognized (MM/DD/YY)
  - "Flights on 15-03-2024" → "15-03-2024" format not recognized (DD-MM-YYYY)
  - "Find flights on 15.03.2024" → "15.03.2024" format not recognized (DD.MM.YYYY)
  - "Show flights on Mar 15, 2024" → "Mar 15, 2024" format not recognized (abbreviated month)
  - "Flights on 15th March 2024" → "15th March 2024" format not recognized (ordinal date)
  - "Find flights on next Monday" → "next Monday" format not recognized (relative date)
  - "Show flights tomorrow" → "tomorrow" format not recognized (relative date)

**Currently Supported Formats**:
- `YYYY-MM-DD` (e.g., "2024-03-15")
- `MM/DD/YYYY` (e.g., "03/15/2024")
- Full month name format (e.g., "January 15, 2024")

### 4. Missing Entities
- **Error Rate**: ~0% (acceptable for general queries)
- **Description**: Entities expected but not extracted (may be acceptable for general queries)
- **Examples**:
  - "Show me flights" → No airports extracted (acceptable for general query)
  - "What are the delays?" → No flight numbers (acceptable for general query)
  - "Compare satisfaction" → No satisfaction threshold (acceptable for general query)
  - "Show routes" → No route airports (acceptable for general query)
  - "Find journeys" → No journey IDs (acceptable for general query)
  - "Tell me about passengers" → No passenger IDs (acceptable for general query)

**Note**: Missing entities are often acceptable for general queries that don't specify particular entities.

## Summary Statistics

| Error Category | Error Rate | Impact |
|---------------|------------|--------|
| Ambiguous queries | ~10% | Medium - May route to wrong retrieval strategy |
| Unrecognized intents | ~95% | High - Many queries fall back to general_question |
| Multi-intent queries | N/A | Medium - Not supported, only one intent returned |
| False positives | ~0% | Low - Effectively handled by exclusion list |
| False negatives | ~10.5% | Medium - Valid airport codes missed |
| Date format variations | ~50% | Medium - Many date formats not recognized |
| Missing entities | ~0% | Low - Often acceptable for general queries |

## Recommendations

1. **Expand Intent Patterns**: Add patterns for booking, cancellation, check-in, and other common airline operations
2. **Expand Airport Code Whitelist**: Add more international airport codes to reduce false negatives
3. **Support More Date Formats**: Add patterns for DD/MM/YY, relative dates, and abbreviated months
4. **Multi-Intent Detection**: Implement support for queries with multiple intents
5. **Context-Aware Classification**: Use query context to better disambiguate ambiguous queries

