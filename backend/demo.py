#!/usr/bin/env python3
"""
NOESIS Demo Script
Demonstrates the complete OSINT civil unrest detection system
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🚨 {title}")
    print(f"{'='*60}")

def print_section(title):
    print(f"\n📋 {title}")
    print("-" * 40)

def demo_collection():
    """Demonstrate data collection capabilities"""
    print_section("Data Collection Pipeline")
    
    print("🔄 Triggering data collection cycle...")
    response = requests.post(f"{BASE_URL}/collection/run-cycle")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Collection completed!")
        print(f"   📊 Raw posts collected: {data.get('raw_posts', 0)}")
        print(f"   🔍 Processed posts: {data.get('processed_posts', 0)}")
        print(f"   🚨 Incidents created: {data.get('incidents', 0)}")
    else:
        print(f"❌ Collection failed: {response.status_code}")

def demo_incidents():
    """Demonstrate incident detection and retrieval"""
    print_section("Incident Detection & Analysis")
    
    # Get dashboard data
    response = requests.get(f"{BASE_URL}/incidents/dashboard")
    if response.status_code == 200:
        dashboard = response.json()
        summary = dashboard['summary']
        
        print(f"📈 System Statistics:")
        print(f"   Total incidents: {summary['total_incidents']}")
        print(f"   Verified incidents: {summary['verified_incidents']}")
        print(f"   High severity: {summary['high_severity_incidents']}")
        print(f"   Verification rate: {summary['verification_rate']:.1f}%")
        
        # Show recent activity
        print(f"\n🆕 Recent Activity:")
        for incident in dashboard['recent_activity'][:3]:
            print(f"   • {incident['title']}")
            print(f"     Location: {incident['location']}")
            print(f"     Severity: {incident['severity']} | Status: {incident['status']}")
            print(f"     Sources: {incident['sources_count']}")
        
        # Show severity distribution
        print(f"\n📊 Severity Distribution:")
        for severity, count in dashboard['severity_distribution'].items():
            print(f"   {severity.title()}: {count}")
    
    # Get latest verified incidents
    print(f"\n🔍 Latest Verified Incidents:")
    response = requests.get(f"{BASE_URL}/incidents/latest?limit=3")
    if response.status_code == 200:
        incidents = response.json()
        for incident in incidents:
            print(f"   • {incident['title']}")
            print(f"     Location: {incident['location']}")
            print(f"     Severity: {incident['severity']}")

def demo_filtering():
    """Demonstrate filtering capabilities"""
    print_section("Advanced Filtering & Search")
    
    # Filter by severity
    print("🔍 Filtering by High Severity:")
    response = requests.get(f"{BASE_URL}/incidents/?severity=high&limit=3")
    if response.status_code == 200:
        incidents = response.json()
        for incident in incidents:
            print(f"   • {incident['title']}")
    
    # Filter by coordinates (using the coordinate-based location)
    print(f"\n🗺️  Filtering by Location (using coordinates):")
    response = requests.get(f"{BASE_URL}/incidents/?region=34.02&limit=2")
    if response.status_code == 200:
        incidents = response.json()
        for incident in incidents:
            print(f"   • {incident['title']}")

def demo_alerts():
    """Demonstrate alert system"""
    print_section("Real-time Alert System")
    
    # Subscribe to alerts
    subscription_data = {
        "email": "demo@noesis.com",
        "region_of_interest": "Global",
        "severity_preference": "high",
        "digest_mode": False
    }
    
    print("📧 Subscribing to alerts...")
    response = requests.post(f"{BASE_URL}/alerts/subscribe", json=subscription_data)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ {result['message']}")
        print(f"   Subscriber ID: {result['subscriber_id']}")
    
    # Show subscribers
    print(f"\n👥 Current Subscribers:")
    response = requests.get(f"{BASE_URL}/alerts/subscribers")
    if response.status_code == 200:
        subscribers = response.json()
        for sub in subscribers:
            print(f"   • {sub['email']} ({sub['severity_preference']} severity)")

def demo_api_endpoints():
    """Demonstrate all API endpoints"""
    print_section("API Endpoints Overview")
    
    endpoints = [
        ("GET", "/incidents/", "Get all incidents"),
        ("GET", "/incidents/latest", "Get latest verified incidents"),
        ("GET", "/incidents/dashboard", "Get real-time dashboard"),
        ("GET", "/incidents/stats/summary", "Get incident statistics"),
        ("POST", "/collection/run-cycle", "Trigger data collection"),
        ("GET", "/collection/status", "Get collection status"),
        ("POST", "/alerts/subscribe", "Subscribe to alerts"),
        ("POST", "/alerts/unsubscribe", "Unsubscribe from alerts"),
        ("GET", "/alerts/subscribers", "Get all subscribers")
    ]
    
    for method, endpoint, description in endpoints:
        print(f"   {method} {endpoint}")
        print(f"      {description}")

def main():
    """Run the complete NOESIS demo"""
    print_header("NOESIS - Real-time OSINT Civil Unrest Detection System")
    print(f"🎯 Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 API Base URL: {BASE_URL}")
    
    try:
        # Test API connectivity
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            print("❌ Cannot connect to NOESIS API. Make sure the server is running.")
            return
        
        # Run demo sections
        demo_api_endpoints()
        demo_collection()
        time.sleep(2)  # Give collection time to process
        demo_incidents()
        demo_filtering()
        demo_alerts()
        
        print_header("Demo Complete!")
        print("🎉 NOESIS system is fully operational!")
        print("💡 Key Features Demonstrated:")
        print("   • Multi-source data collection (Twitter, Reddit, News)")
        print("   • AI-powered incident detection and verification")
        print("   • Real-time alert system")
        print("   • Advanced filtering and search")
        print("   • Comprehensive REST API")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to NOESIS API. Please start the server with:")
        print("   python start.py")
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    main() 