#!/usr/bin/env python3
"""
Quick Data Refresh Script
Simple script to refresh data and update predictions
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.simple_collector import SimpleRealTimeCollector

async def quick_refresh():
    """Quick data refresh"""
    print("🔄 Starting quick data refresh...")
    start_time = datetime.now()
    
    try:
        # Collect fresh data
        collector = SimpleRealTimeCollector()
        data = await collector.collect_all_data()
        
        print(f"✅ Collected {len(data)} data points")
        print(f"📊 Data sources: {[item.source for item in data]}")
        
        # Show sample data
        if data:
            print("\n📋 Sample data:")
            for i, item in enumerate(data[:3]):  # Show first 3 items
                print(f"  {i+1}. {item.source}: {item.content[:100]}...")
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\n🎉 Refresh completed in {duration}")
        
    except Exception as e:
        print(f"❌ Refresh failed: {e}")

if __name__ == "__main__":
    asyncio.run(quick_refresh()) 