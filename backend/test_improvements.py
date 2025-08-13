#!/usr/bin/env python3
"""
Test script to verify the improved filtering and severity classification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.nlp_pipeline import NLPPipeline
from app.services.verification import VerificationService
from datetime import datetime, timedelta

def test_relevance_filtering():
    """Test the improved protest relevance classification"""
    print("=" * 60)
    print("TESTING IMPROVED RELEVANCE FILTERING")
    print("=" * 60)
    
    nlp = NLPPipeline()
    
    # Test cases - should be filtered OUT (irrelevant)
    irrelevant_texts = [
        "Manchester United beats Chelsea 3-1 in Premier League match today",
        "New iPhone 15 released with improved camera and battery life",
        "Stock market sees gains as tech shares rise 5% today",
        "Weather forecast: Heavy rain expected this weekend",
        "Celebrity couple announces engagement at movie premiere",
        "Recipe: How to make the perfect chocolate chip cookies",
        "Bitcoin price surges to new monthly high amid investor optimism"
    ]
    
    # Test cases - should be included (relevant)
    relevant_texts = [
        "Protest erupts in downtown area as police arrest demonstrators",
        "Civil unrest reported in multiple cities following controversial verdict",
        "Strike organized by workers demanding better wages and conditions",
        "Clash between protesters and police results in several injuries",
        "Demonstration against government policy turns violent",
        "Rally for civil rights draws thousands to city center",
        "Riot police deploy tear gas as unrest escalates"
    ]
    
    print("\nTesting IRRELEVANT texts (should get low scores):")
    for text in irrelevant_texts:
        score = nlp.classify_protest_relevance(text)
        status = "✅ FILTERED" if score < 0.3 else "❌ NOT FILTERED"
        print(f"{status} Score: {score:.3f} - {text[:60]}...")
    
    print("\nTesting RELEVANT texts (should get high scores):")
    for text in relevant_texts:
        score = nlp.classify_protest_relevance(text)
        status = "✅ INCLUDED" if score >= 0.3 else "❌ EXCLUDED"
        print(f"{status} Score: {score:.3f} - {text[:60]}...")

def test_severity_classification():
    """Test the improved severity classification"""
    print("\n" + "=" * 60)
    print("TESTING IMPROVED SEVERITY CLASSIFICATION")
    print("=" * 60)
    
    verification = VerificationService()
    
    # Test cases for different severity levels
    test_clusters = [
        {
            "name": "HIGH SEVERITY",
            "cluster": [
                {
                    "id": 1,
                    "protest_score": 0.8,
                    "sentiment_score": -0.7,
                    "content": "Breaking: Multiple people killed in violent clashes as riot police deploy tear gas against protesters. Military called in to restore order.",
                    "title": "Deadly protest turns violent",
                    "platform": "news",
                    "timestamp": datetime.now().isoformat(),
                    "location_lat": 40.7128,
                    "location_lng": -74.0060
                },
                {
                    "id": 2,
                    "protest_score": 0.9,
                    "sentiment_score": -0.8,
                    "content": "State of emergency declared as civil unrest spreads across the city. Thousands injured in ongoing violence.",
                    "title": "State of emergency declared",
                    "platform": "twitter",
                    "timestamp": datetime.now().isoformat(),
                    "location_lat": 40.7128,
                    "location_lng": -74.0060
                }
            ]
        },
        {
            "name": "MEDIUM SEVERITY",
            "cluster": [
                {
                    "id": 3,
                    "protest_score": 0.6,
                    "sentiment_score": -0.4,
                    "content": "Police arrest dozens during peaceful protest that turned confrontational. Several minor injuries reported.",
                    "title": "Arrests made during protest",
                    "platform": "reddit",
                    "timestamp": datetime.now().isoformat(),
                    "location_lat": 40.7128,
                    "location_lng": -74.0060
                },
                {
                    "id": 4,
                    "protest_score": 0.7,
                    "sentiment_score": -0.3,
                    "content": "Demonstration blocks major intersection as protesters demand justice. Police monitoring situation.",
                    "title": "Major intersection blocked",
                    "platform": "news",
                    "timestamp": datetime.now().isoformat(),
                    "location_lat": 40.7128,
                    "location_lng": -74.0060
                }
            ]
        },
        {
            "name": "LOW SEVERITY",
            "cluster": [
                {
                    "id": 5,
                    "protest_score": 0.4,
                    "sentiment_score": -0.1,
                    "content": "Small peaceful gathering in park to discuss community issues. No incidents reported.",
                    "title": "Peaceful community gathering",
                    "platform": "reddit",
                    "timestamp": datetime.now().isoformat(),
                    "location_lat": 40.7128,
                    "location_lng": -74.0060
                }
            ]
        }
    ]
    
    for test_case in test_clusters:
        print(f"\nTesting {test_case['name']}:")
        incident = verification.create_incident(test_case["cluster"])
        severity_indicators = verification._analyze_severity_indicators(test_case["cluster"])
        
        print(f"  Expected: {test_case['name']}")
        print(f"  Actual Severity: {incident['severity'].upper()}")
        print(f"  Status: {incident['status']}")
        print(f"  Confidence: {incident['confidence_score']}")
        print(f"  Severity Indicators:")
        for key, value in severity_indicators.items():
            print(f"    {key}: {value}/4")
        
        expected_severity = test_case['name'].split()[0].lower()
        actual_severity = incident['severity']
        status = "✅ CORRECT" if expected_severity == actual_severity else "❌ INCORRECT"
        print(f"  Result: {status}")

def test_incident_validation():
    """Test the new incident validation logic"""
    print("\n" + "=" * 60)
    print("TESTING INCIDENT VALIDATION")
    print("=" * 60)
    
    verification = VerificationService()
    
    # Test valid incident
    valid_incident = {
        "title": "Major protest in downtown results in multiple arrests",
        "location": "Downtown New York",
        "location_lat": 40.7128,
        "location_lng": -74.0060,
        "confidence_score": 75
    }
    
    # Test invalid incidents
    invalid_incidents = [
        {
            "title": "abc",  # Too short
            "location": "Downtown",
            "confidence_score": 50
        },
        {
            "title": "Some protest happening",
            "location": "Unknown Location",  # Invalid location
            "location_lat": None,
            "location_lng": None,
            "confidence_score": 50
        },
        {
            "title": "Click here for amazing deals! Buy now!",  # Spam
            "location": "Downtown",
            "confidence_score": 50
        },
        {
            "title": "Legitimate protest happening",
            "location": "Downtown",
            "confidence_score": 20  # Low confidence
        }
    ]
    
    print(f"Valid incident: {verification._is_valid_incident(valid_incident)} ✅")
    
    for i, incident in enumerate(invalid_incidents):
        is_valid = verification._is_valid_incident(incident)
        reason = ["too short title", "invalid location", "spam content", "low confidence"][i]
        status = "❌ REJECTED" if not is_valid else "⚠️ ACCEPTED"
        print(f"Invalid incident ({reason}): {is_valid} {status}")

if __name__ == "__main__":
    print("TESTING IMPROVED NOESIS FILTERING AND SEVERITY CLASSIFICATION")
    print("=" * 80)
    
    try:
        test_relevance_filtering()
        test_severity_classification()
        test_incident_validation()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
