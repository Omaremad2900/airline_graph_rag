# Retrieval Layer: New Intent Support

## Summary

All new intent categories added to the preprocessing layer are now supported in the retrieval layer with appropriate Cypher query templates.

## New Intent Templates Added

### 1. `flight_status` ✅
**Purpose**: Check flight status, delays, and on-time performance

**Templates**:
- `flight_status_by_number`: Get status for a specific flight number
- `flight_status_recent`: Get recent journey data for a flight
- `on_time_flights`: Find flights with best on-time performance

**Parameters**:
- `flight_number` (from FLIGHT entity)

**Example Queries**:
- "What is the status of flight AA123?"
- "Is flight DL456 on time?"
- "Show me on-time flights"

### 2. `loyalty` ✅
**Purpose**: Analyze loyalty program data, passenger classes, and frequent flyer information

**Templates**:
- `loyalty_passenger_analysis`: Analyze journeys by passenger class
- `loyalty_by_class`: Statistics by passenger class

**Parameters**: None required

**Example Queries**:
- "Show me frequent flyer journeys"
- "What are the loyalty program statistics?"
- "Compare passenger classes"

### 3. `booking` ✅
**Purpose**: Find available flights for booking

**Templates**:
- `available_flights`: Get available flights (optionally filtered by route)
- `popular_flights`: Get most popular/booked flights

**Parameters**:
- `departure_code` (optional, from AIRPORT entity)
- `arrival_code` (optional, from AIRPORT entity)

**Example Queries**:
- "Book a flight from JFK to LAX"
- "Show me available flights"
- "What are the popular flights?"

### 4. `cancellation` ✅
**Purpose**: Identify flights with high cancellation risk or delay patterns

**Templates**:
- `cancelled_flight_patterns`: Find flights with high delay rates (potential cancellations)
- `flight_reliability`: Analyze flight reliability and high delay percentage

**Parameters**: None required

**Example Queries**:
- "Which flights are often cancelled?"
- "Show me unreliable flights"
- "What flights have high cancellation risk?"

### 5. `check_in` ✅
**Purpose**: Get check-in information for flights

**Templates**:
- `check_in_info`: Get flight and passenger information for check-in

**Parameters**:
- `flight_number` (from FLIGHT entity)

**Example Queries**:
- "Check in for flight AA123"
- "What is the check-in information for flight DL456?"

### 6. `seat_selection` ✅
**Purpose**: Get seat class availability and performance

**Templates**:
- `seat_class_availability`: Get available seat classes for a flight
- `class_performance`: Compare performance across passenger classes

**Parameters**:
- `flight_number` (optional, from FLIGHT entity)

**Example Queries**:
- "What seat classes are available on flight AA123?"
- "Compare seat class performance"
- "Show me seat selection options"

### 7. `baggage` ✅
**Purpose**: Get baggage-related journey information

**Templates**:
- `baggage_related_journeys`: Get journey data that might relate to baggage (satisfaction, class, miles)

**Parameters**: None required

**Example Queries**:
- "What is the baggage allowance?"
- "Show me baggage-related information"

## Intent Support Matrix

| Intent | Templates | Parameters | Status |
|--------|-----------|------------|--------|
| `flight_search` | 3 | AIRPORT | ✅ Existing |
| `delay_analysis` | 3 | NUMBER (optional) | ✅ Existing |
| `passenger_satisfaction` | 3 | NUMBER (optional) | ✅ Existing |
| `route_analysis` | 3 | None | ✅ Existing |
| `journey_insights` | 2 | JOURNEY (optional) | ✅ Existing |
| `performance_metrics` | 2 | None | ✅ Existing |
| `recommendation` | 1 | None | ✅ Existing |
| `general_question` | 1 | FLIGHT | ✅ Existing |
| `flight_status` | 3 | FLIGHT | ✅ **NEW** |
| `loyalty` | 2 | None | ✅ **NEW** |
| `booking` | 2 | AIRPORT (optional) | ✅ **NEW** |
| `cancellation` | 2 | None | ✅ **NEW** |
| `check_in` | 1 | FLIGHT | ✅ **NEW** |
| `seat_selection` | 2 | FLIGHT (optional) | ✅ **NEW** |
| `baggage` | 1 | None | ✅ **NEW** |

## Fallback Behavior

If an intent is not recognized or has no matching templates, the system:
1. Logs a warning message
2. Falls back to `general_question` intent
3. Uses the `flight_info` template (if flight number is available)

This ensures graceful degradation for unrecognized intents.

## Testing

All new intents have been added to the query templates dictionary. The retrieval layer will now:
- ✅ Recognize all new intent categories
- ✅ Execute appropriate Cypher queries
- ✅ Return relevant results from Neo4j
- ✅ Handle missing parameters gracefully

## Notes

1. **Transactional Operations**: Some intents like `booking` and `cancellation` are typically transactional operations in real systems. The current implementation provides informational queries (available flights, cancellation patterns) rather than actual booking/cancellation functionality.

2. **Data Availability**: Some queries may return limited results if the Neo4j graph doesn't contain the relevant data (e.g., specific baggage information, seat selection details).

3. **Parameter Handling**: The existing `_build_parameters` method automatically extracts entities (FLIGHT, AIRPORT, etc.) and maps them to query parameters, so the new intents work seamlessly with the existing entity extraction system.

