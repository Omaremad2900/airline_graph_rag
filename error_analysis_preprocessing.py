"""Error Analysis for Input Preprocessing Pipeline.

This script analyzes errors in:
1. Intent Classification (ambiguous queries, unrecognized intents, multi-intent)
2. Entity Extraction (false positives, false negatives, date formats, missing entities)
"""

import json
from typing import Dict, List, Tuple
from preprocessing.intent_classifier import IntentClassifier
from preprocessing.entity_extractor import EntityExtractor


class PreprocessingErrorAnalyzer:
    """Analyzes errors in the preprocessing pipeline."""
    
    def __init__(self):
        self.classifier = IntentClassifier()
        self.extractor = EntityExtractor()
        self.error_report = {
            "intent_classification": {
                "ambiguous_queries": [],
                "unrecognized_intents": [],
                "multi_intent_queries": []
            },
            "entity_extraction": {
                "false_positives": [],
                "false_negatives": [],
                "date_format_issues": [],
                "missing_entities": []
            }
        }
    
    def analyze_intent_classification(self) -> Dict:
        """Analyze intent classification errors."""
        print("=" * 80)
        print("INTENT CLASSIFICATION ERROR ANALYSIS")
        print("=" * 80)
        
        # Test cases for ambiguous queries
        ambiguous_test_cases = [
            ("Show me flights", ["flight_search", "general_question"]),
            ("What flights?", ["flight_search", "general_question"]),
            ("Flights", ["flight_search", "general_question"]),
            ("Tell me about delays", ["delay_analysis", "general_question"]),
            ("What about satisfaction?", ["passenger_satisfaction", "general_question"]),
            ("Routes", ["route_analysis", "general_question"]),
            ("Performance", ["performance_metrics", "general_question"]),
            ("Compare", ["performance_metrics", "recommendation", "general_question"]),
            ("Best flights", ["flight_search", "recommendation", "general_question"]),
            ("Flight information", ["flight_search", "general_question"]),
        ]
        
        ambiguous_count = 0
        total_ambiguous = len(ambiguous_test_cases)
        
        print("\n1. AMBIGUOUS QUERIES (Multiple valid intents):")
        print("-" * 80)
        for query, possible_intents in ambiguous_test_cases:
            predicted = self.classifier.classify(query)
            if predicted not in possible_intents:
                ambiguous_count += 1
                self.error_report["intent_classification"]["ambiguous_queries"].append({
                    "query": query,
                    "predicted": predicted,
                    "possible_intents": possible_intents,
                    "status": "MISCLASSIFIED"
                })
                print(f"[ERROR] Query: '{query}'")
                print(f"   Predicted: {predicted}")
                print(f"   Possible: {possible_intents}")
            else:
                self.error_report["intent_classification"]["ambiguous_queries"].append({
                    "query": query,
                    "predicted": predicted,
                    "possible_intents": possible_intents,
                    "status": "CORRECT (but ambiguous)"
                })
                print(f"[WARNING] Query: '{query}'")
                print(f"   Predicted: {predicted} (one of {possible_intents})")
            print()
        
        ambiguous_rate = (ambiguous_count / total_ambiguous) * 100
        print(f"Ambiguous Query Misclassification Rate: {ambiguous_rate:.1f}% ({ambiguous_count}/{total_ambiguous})")
        
        # Test cases for unrecognized intents (queries that don't match any pattern)
        unrecognized_test_cases = [
            "Book a flight",
            "Cancel my reservation",
            "Check in online",
            "Upgrade my seat",
            "Change my flight",
            "Refund policy",
            "Baggage allowance",
            "Check flight status",
            "Weather at airport",
            "Airport parking",
            "Lounge access",
            "Frequent flyer miles",
            "Flight cancellation",
            "Seat selection",
            "In-flight entertainment",
            "Meal preferences",
            "Special assistance",
            "Travel insurance",
            "Visa requirements",
            "Currency exchange"
        ]
        
        unrecognized_count = 0
        total_unrecognized = len(unrecognized_test_cases)
        
        print("\n2. UNRECOGNIZED INTENTS (Falls back to general_question):")
        print("-" * 80)
        for query in unrecognized_test_cases:
            predicted = self.classifier.classify(query)
            if predicted == "general_question":
                unrecognized_count += 1
                self.error_report["intent_classification"]["unrecognized_intents"].append({
                    "query": query,
                    "predicted": predicted,
                    "status": "UNRECOGNIZED"
                })
                print(f"[ERROR] Query: '{query}' -> {predicted} (fallback)")
            else:
                print(f"[OK] Query: '{query}' -> {predicted}")
        
        unrecognized_rate = (unrecognized_count / total_unrecognized) * 100
        print(f"\nUnrecognized Intent Rate: {unrecognized_rate:.1f}% ({unrecognized_count}/{total_unrecognized})")
        
        # Test cases for multi-intent queries (not supported)
        multi_intent_test_cases = [
            "Compare delays and satisfaction",
            "Show me flights and routes",
            "What are delays and ratings?",
            "Compare performance and satisfaction",
            "Find flights and check delays",
            "Show routes and passenger feedback",
            "Compare delays with satisfaction ratings",
            "Find flights with good ratings",
            "Show delays and recommend best routes",
            "Compare routes and performance metrics"
        ]
        
        print("\n3. MULTI-INTENT QUERIES (Not supported - only one intent returned):")
        print("-" * 80)
        for query in multi_intent_test_cases:
            predicted = self.classifier.classify(query)
            self.error_report["intent_classification"]["multi_intent_queries"].append({
                "query": query,
                "predicted": predicted,
                "status": "SINGLE_INTENT_ONLY"
            })
            print(f"[WARNING] Query: '{query}'")
            print(f"   Predicted: {predicted} (only one intent, multi-intent not supported)")
            print()
        
        print(f"Multi-intent queries tested: {len(multi_intent_test_cases)}")
        print("Status: Multi-intent detection not implemented")
        
        return {
            "ambiguous_misclassification_rate": ambiguous_rate,
            "unrecognized_intent_rate": unrecognized_rate,
            "multi_intent_support": False
        }
    
    def analyze_entity_extraction(self) -> Dict:
        """Analyze entity extraction errors."""
        print("\n" + "=" * 80)
        print("ENTITY EXTRACTION ERROR ANALYSIS")
        print("=" * 80)
        
        # Test cases for false positives (incorrectly extracted entities)
        false_positive_cases = [
            ("Find flights from THE airport", "THE", "AIRPORT", "Common word 'THE' mistaken for airport code"),
            ("Show me flights from AND to LAX", "AND", "AIRPORT", "Common word 'AND' mistaken for airport code"),
            ("Flights from FOR airport", "FOR", "AIRPORT", "Common word 'FOR' mistaken for airport code"),
            ("What about flights from ARE airport", "ARE", "AIRPORT", "Common word 'ARE' mistaken for airport code"),
            ("Show flights from ALL airports", "ALL", "AIRPORT", "Common word 'ALL' mistaken for airport code"),
        ]
        
        false_positive_count = 0
        total_fp = len(false_positive_cases)
        
        print("\n1. FALSE POSITIVES (Incorrectly extracted entities):")
        print("-" * 80)
        for query, false_entity, entity_type, description in false_positive_cases:
            entities = self.extractor.extract_entities(query)
            found_false = False
            
            if entity_type in entities:
                for entity in entities[entity_type]:
                    if entity["value"].upper() == false_entity.upper():
                        found_false = True
                        false_positive_count += 1
                        self.error_report["entity_extraction"]["false_positives"].append({
                            "query": query,
                            "false_entity": false_entity,
                            "entity_type": entity_type,
                            "description": description,
                            "extracted_entities": entities
                        })
                        print(f"[ERROR] {description}")
                        print(f"   Query: '{query}'")
                        print(f"   False entity extracted: {false_entity}")
                        print(f"   All entities: {json.dumps(entities, indent=2)}")
                        break
            
            if not found_false:
                print(f"[OK] Query: '{query}' - No false positive detected")
            print()
        
        false_positive_rate = (false_positive_count / total_fp) * 100
        print(f"False Positive Rate: {false_positive_rate:.1f}% ({false_positive_count}/{total_fp})")
        
        # Test cases for false negatives (missed valid entities)
        false_negative_cases = [
            ("Show flights from XYZ to ABC", ["XYZ", "ABC"], "AIRPORT", "Valid airport codes not in whitelist"),
            ("Find flights from BKK to KUL", ["BKK", "KUL"], "AIRPORT", "International codes should be recognized"),
            ("Flights from DXB to IST", ["DXB", "IST"], "AIRPORT", "International codes should be recognized"),
            ("Show flights from SIN to ICN", ["SIN", "ICN"], "AIRPORT", "International codes should be recognized"),
            ("Find flights from PEK to PVG", ["PEK", "PVG"], "AIRPORT", "International codes should be recognized"),
            ("Flights from CAN airport", ["CAN"], "AIRPORT", "International code should be recognized"),
            ("Show flights from MAD to BCN", ["MAD", "BCN"], "AIRPORT", "International codes should be recognized"),
            ("Find flights from FCO to MUC", ["FCO", "MUC"], "AIRPORT", "International codes should be recognized"),
            ("Flights from ZUR to VIE", ["ZUR", "VIE"], "AIRPORT", "International codes should be recognized"),
            ("Show flights from CPH to OSL", ["CPH", "OSL"], "AIRPORT", "International codes should be recognized"),
        ]
        
        false_negative_count = 0
        total_expected = 0
        
        print("\n2. FALSE NEGATIVES (Missed valid entities):")
        print("-" * 80)
        for query, expected_entities, entity_type, description in false_negative_cases:
            entities = self.extractor.extract_entities(query)
            extracted_values = []
            
            if entity_type in entities:
                extracted_values = [e["value"].upper() for e in entities[entity_type]]
            
            missed = []
            for expected in expected_entities:
                total_expected += 1
                if expected.upper() not in extracted_values:
                    missed.append(expected)
                    false_negative_count += 1
            
            if missed:
                self.error_report["entity_extraction"]["false_negatives"].append({
                    "query": query,
                    "expected_entities": expected_entities,
                    "missed_entities": missed,
                    "entity_type": entity_type,
                    "description": description,
                    "extracted_entities": entities
                })
                print(f"[ERROR] {description}")
                print(f"   Query: '{query}'")
                print(f"   Expected: {expected_entities}")
                print(f"   Missed: {missed}")
                print(f"   Extracted: {extracted_values}")
            else:
                print(f"[OK] Query: '{query}' - All entities found: {extracted_values}")
            print()
        
        false_negative_rate = (false_negative_count / total_expected) * 100 if total_expected > 0 else 0
        print(f"False Negative Rate: {false_negative_rate:.1f}% ({false_negative_count}/{total_expected} expected entities missed)")
        
        # Test cases for date format variations
        date_format_cases = [
            ("Find flights on 15/03/24", "15/03/24", "DD/MM/YY format not recognized"),
            ("Show flights on 03/15/24", "03/15/24", "MM/DD/YY format not recognized"),
            ("Flights on 15-03-2024", "15-03-2024", "DD-MM-YYYY format not recognized"),
            ("Find flights on 15.03.2024", "15.03.2024", "DD.MM.YYYY format not recognized"),
            ("Show flights on Mar 15, 2024", "Mar 15, 2024", "Abbreviated month format not recognized"),
            ("Flights on 15th March 2024", "15th March 2024", "Ordinal date format not recognized"),
            ("Find flights on next Monday", "next Monday", "Relative date not recognized"),
            ("Show flights tomorrow", "tomorrow", "Relative date not recognized"),
        ]
        
        date_format_issues = 0
        total_date_cases = len(date_format_cases)
        
        print("\n3. DATE FORMAT VARIATIONS (Unrecognized formats):")
        print("-" * 80)
        for query, date_str, description in date_format_cases:
            entities = self.extractor.extract_entities(query)
            date_found = False
            
            if "DATE" in entities:
                for date_entity in entities["DATE"]:
                    if date_str.lower() in date_entity["value"].lower() or date_entity["value"].lower() in date_str.lower():
                        date_found = True
                        break
            
            if not date_found:
                date_format_issues += 1
                self.error_report["entity_extraction"]["date_format_issues"].append({
                    "query": query,
                    "date_string": date_str,
                    "description": description,
                    "extracted_entities": entities
                })
                print(f"[ERROR] {description}")
                print(f"   Query: '{query}'")
                print(f"   Date string: '{date_str}' not recognized")
            else:
                print(f"[OK] Query: '{query}' - Date recognized")
            print()
        
        date_format_rate = (date_format_issues / total_date_cases) * 100
        print(f"Unrecognized Date Format Rate: {date_format_rate:.1f}% ({date_format_issues}/{total_date_cases})")
        
        # Test cases for missing entities (entities that should be extracted but aren't)
        missing_entity_cases = [
            ("Show me flights", [], "AIRPORT", "No airports extracted from flight query"),
            ("What are the delays?", [], "FLIGHT", "No flight numbers in delay query"),
            ("Compare satisfaction", [], "NUMBER", "No satisfaction threshold extracted"),
            ("Show routes", [], "AIRPORT", "No route airports extracted"),
            ("Find journeys", [], "JOURNEY", "No journey IDs extracted"),
            ("Tell me about passengers", [], "PASSENGER", "No passenger IDs extracted"),
        ]
        
        missing_count = 0
        total_missing = len(missing_entity_cases)
        
        print("\n4. MISSING ENTITIES (Entities expected but not extracted):")
        print("-" * 80)
        for query, expected_entities, entity_type, description in missing_entity_cases:
            entities = self.extractor.extract_entities(query)
            
            # Check if entity type is missing or empty
            if entity_type not in entities or len(entities.get(entity_type, [])) == 0:
                if expected_entities:  # Only count if entities were actually expected
                    missing_count += 1
                    self.error_report["entity_extraction"]["missing_entities"].append({
                        "query": query,
                        "expected_entity_type": entity_type,
                        "expected_entities": expected_entities,
                        "description": description,
                        "extracted_entities": entities
                    })
                    print(f"[WARNING] {description}")
                    print(f"   Query: '{query}'")
                    print(f"   Expected: {entity_type}")
                    print(f"   Extracted: {json.dumps(entities, indent=2)}")
                else:
                    # Entity type not present but not necessarily expected
                    print(f"[INFO] Query: '{query}' - {entity_type} not extracted (may be acceptable)")
            else:
                print(f"[OK] Query: '{query}' - {entity_type} found: {entities[entity_type]}")
            print()
        
        missing_rate = (missing_count / total_missing) * 100
        print(f"Missing Entity Rate: {missing_rate:.1f}% ({missing_count}/{total_missing})")
        
        return {
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "date_format_issue_rate": date_format_rate,
            "missing_entity_rate": missing_rate
        }
    
    def generate_summary_report(self, intent_stats: Dict, entity_stats: Dict):
        """Generate a summary error report."""
        print("\n" + "=" * 80)
        print("ERROR ANALYSIS SUMMARY")
        print("=" * 80)
        
        print("\nINTENT CLASSIFICATION ERRORS:")
        print(f"  • Ambiguous queries misclassification: ~{intent_stats['ambiguous_misclassification_rate']:.1f}%")
        print(f"  • Unrecognized intents (fallback to general_question): ~{intent_stats['unrecognized_intent_rate']:.1f}%")
        print(f"  • Multi-intent queries: Not supported")
        
        print("\nENTITY EXTRACTION ERRORS:")
        print(f"  • False positives: ~{entity_stats['false_positive_rate']:.1f}%")
        print(f"  • False negatives: ~{entity_stats['false_negative_rate']:.1f}%")
        print(f"  • Date format variations: ~{entity_stats['date_format_issue_rate']:.1f}%")
        print(f"  • Missing entities: ~{entity_stats['missing_entity_rate']:.1f}%")
        
        print("\n" + "=" * 80)
        print("DETAILED EXAMPLES")
        print("=" * 80)
        
        print("\n1. Ambiguous Query Examples:")
        for item in self.error_report["intent_classification"]["ambiguous_queries"][:3]:
            print(f"   - \"{item['query']}\" -> Could be {item['possible_intents']}")
        
        print("\n2. Unrecognized Intent Examples:")
        for item in self.error_report["intent_classification"]["unrecognized_intents"][:3]:
            print(f"   - \"{item['query']}\" -> Falls back to {item['predicted']}")
        
        print("\n3. False Positive Examples:")
        for item in self.error_report["entity_extraction"]["false_positives"][:2]:
            print(f"   - \"{item['query']}\" -> \"{item['false_entity']}\" incorrectly extracted")
        
        print("\n4. False Negative Examples:")
        for item in self.error_report["entity_extraction"]["false_negatives"][:3]:
            print(f"   - \"{item['query']}\" -> \"{', '.join(item['missed_entities'])}\" not recognized")
        
        print("\n5. Date Format Examples:")
        for item in self.error_report["entity_extraction"]["date_format_issues"][:3]:
            print(f"   - \"{item['query']}\" -> \"{item['date_string']}\" format not recognized")
        
        print("\n6. Missing Entity Examples:")
        for item in self.error_report["entity_extraction"]["missing_entities"][:3]:
            print(f"   - \"{item['query']}\" -> {item['expected_entity_type']} expected but not found")
    
    def save_report(self, filename: str = "preprocessing_error_report.json"):
        """Save detailed error report to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.error_report, f, indent=2)
        print(f"\nDetailed error report saved to: {filename}")


def main():
    """Run the error analysis."""
    analyzer = PreprocessingErrorAnalyzer()
    
    # Analyze intent classification
    intent_stats = analyzer.analyze_intent_classification()
    
    # Analyze entity extraction
    entity_stats = analyzer.analyze_entity_extraction()
    
    # Generate summary
    analyzer.generate_summary_report(intent_stats, entity_stats)
    
    # Save detailed report
    analyzer.save_report()
    
    print("\n[SUCCESS] Error analysis complete!")


if __name__ == "__main__":
    main()

