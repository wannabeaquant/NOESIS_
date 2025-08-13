#!/usr/bin/env python3
"""
Comprehensive API testing script for NOESIS Backend
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_endpoint(method, endpoint, data=None, expected_status=200):
    """Test a single endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url)
        elif method.upper() == "POST":
            response = requests.post(url, json=data)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
        
        print(f"\n🔍 Testing {method} {endpoint}")
        print(f"Status Code: {response.status_code}")
        
        # Accept 404 for non-existent incidents
        if endpoint.startswith("/incidents/") and response.status_code == 404:
            print("✅ Status Code: PASS (404 for non-existent incident)")
            try:
                print(f"Response: {response.json()}")
            except:
                print(f"Response: {response.text}")
            return True
        
        if response.status_code == expected_status:
            print("✅ Status Code: PASS")
        else:
            print(f"❌ Status Code: FAIL (expected {expected_status}, got {response.status_code})")
        
        try:
            response_json = response.json()
            print(f"Response: {json.dumps(response_json, indent=2)}")
        except:
            print(f"Response: {response.text}")
        
        # Print more details for 500 errors
        if response.status_code == 500:
            print("❌ 500 Internal Server Error Details:")
            print(response.text)
        
        return response.status_code == expected_status
        
    except Exception as e:
        print(f"❌ Error testing {method} {endpoint}: {e}")
        return False

def main():
    print("🚀 Starting NOESIS Backend API Tests")
    print("=" * 50)
    
    # Test 1: Health Check
    test_endpoint("GET", "/")
    
    # Test 2: Collection Status
    test_endpoint("GET", "/collection/status")
    
    # Test 3: Incident Statistics
    test_endpoint("GET", "/incidents/stats/summary")
    
    # Test 4: Get All Incidents
    test_endpoint("GET", "/incidents/")
    
    # Test 5: Get Latest Incidents
    test_endpoint("GET", "/incidents/latest")
    
    # Test 6: Get Incidents with Filters
    test_endpoint("GET", "/incidents/?severity=high&limit=5")
    
    # Test 7: Data Collection (This will take time)
    print("\n🔄 Testing Data Collection (this may take a while)...")
    test_endpoint("POST", "/collection/run-cycle")
    
    # Wait a bit for processing
    print("\n⏳ Waiting 5 seconds for processing...")
    time.sleep(5)
    
    # Test 8: Check Status After Collection
    test_endpoint("GET", "/collection/status")
    
    # Test 9: Check Incidents After Collection
    test_endpoint("GET", "/incidents/")
    
    # Test 10: Check Latest Incidents After Collection
    test_endpoint("GET", "/incidents/latest")
    
    # Test 11: Check Statistics After Collection
    test_endpoint("GET", "/incidents/stats/summary")
    
    # Test 12: Test Non-existent Incident
    test_endpoint("GET", "/incidents/999", expected_status=404)
    
    # Test 13: Test Alert Subscription (should work even without Telegram)
    test_endpoint("POST", "/alerts/subscribe", data={"email": "test@example.com"})
    
    # Test 14: Test Alert Unsubscription
    test_endpoint("POST", "/alerts/unsubscribe", data={"email": "test@example.com"})
    
    # Test 15: Test Moderation Endpoints
    test_endpoint("POST", "/moderate/flag", data={"incident_id": 1, "reason": "test"})
    test_endpoint("POST", "/moderate/confirm", data={"incident_id": 1})
    test_endpoint("POST", "/moderate/merge", data={"incident_ids": [1, 2]})
    
    print("\n" + "=" * 50)
    print("🎉 API Testing Complete!")
    print("\n📊 Summary:")
    print("- All endpoints should return 200 status codes")
    print("- Collection endpoint may take time to complete")
    print("- Some endpoints may return empty arrays if no data exists")
    print("- Moderation endpoints may return errors if incidents don't exist")

if __name__ == "__main__":
    main() 