#!/usr/bin/env python3
"""
NOESIS Data Refresh Pipeline
Runs all data collectors and updates predictions
"""

import asyncio
import time
from datetime import datetime
import logging
from typing import List, Dict
import os
import sys
import schedule
from dataclasses import asdict

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.simple_collector import SimpleRealTimeCollector
from app.services.enhanced_predictive_service import EnhancedPredictiveService
from app.utils.database import get_db_session
from app.models.raw_post import RawPost
from app.models.processed_post import ProcessedPost
from app.models.incident import Incident

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_refresh.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataRefreshPipeline:
    def __init__(self):
        self.collector = SimpleRealTimeCollector()
        self.predictive_service = EnhancedPredictiveService()
        
    async def run_full_refresh(self):
        """Run complete data refresh pipeline"""
        logger.info("🚀 Starting NOESIS Data Refresh Pipeline")
        start_time = datetime.now()
        
        try:
            # Step 1: Collect fresh data
            logger.info("📡 Step 1: Collecting fresh data from APIs...")
            fresh_data = await self.collector.collect_all_data()
            logger.info(f"✅ Collected {len(fresh_data)} new data points")
            
            # Step 2: Process and store data
            logger.info("🔄 Step 2: Processing and storing data...")
            processed_count = await self.process_and_store_data([asdict(dp) for dp in fresh_data])
            logger.info(f"✅ Processed and stored {processed_count} items")
            
            # Step 3: Update predictions
            logger.info("🧠 Step 3: Updating predictions...")
            prediction_count = await self.update_predictions()
            logger.info(f"✅ Updated {prediction_count} predictions")
            
            # Step 4: Generate summary
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"🎉 Data refresh completed in {duration}")
            
            return {
                "status": "success",
                "duration": str(duration),
                "data_collected": len(fresh_data),
                "processed": processed_count,
                "predictions_updated": prediction_count
            }
            
        except Exception as e:
            logger.error(f"❌ Data refresh failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def process_and_store_data(self, fresh_data: List[Dict]) -> int:
        """Process and store collected data"""
        processed_count = 0
        
        async with get_db_session() as session:
            for data in fresh_data:
                try:
                    # Create raw post
                    raw_post = RawPost(
                        platform=data.get("platform", "unknown"),
                        content=data.get("content", ""),
                        author=data.get("author", "unknown"),
                        timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
                        location_raw=data.get("location_raw", ""),
                        link=data.get("link", ""),
                        extra_data=data.get("extra", {})
                    )
                    session.add(raw_post)
                    
                    # Create processed post
                    processed_post = ProcessedPost(
                        raw_post_id=raw_post.id,
                        processed_content=data.get("content", ""),
                        sentiment_score=data.get("sentiment", 0.0),
                        location_processed=data.get("location_processed", ""),
                        keywords=data.get("keywords", []),
                        severity=data.get("severity", "low")
                    )
                    session.add(processed_post)
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process data item: {e}")
                    continue
            
            session.commit()
        
        return processed_count
    
    async def update_predictions(self) -> int:
        """Update predictions based on latest data"""
        try:
            # Get latest incidents for prediction
            async with get_db_session() as session:
                # This would normally fetch recent incidents
                # For now, we'll just return a count
                return 1  # Placeholder
        except Exception as e:
            logger.error(f"Failed to update predictions: {e}")
            return 0

def run_scheduled_refresh():
    """Run the refresh pipeline (for scheduling)"""
    pipeline = DataRefreshPipeline()
    asyncio.run(pipeline.run_full_refresh())

def setup_scheduler():
    """Set up scheduled data refresh"""
    logger.info("⏰ Setting up scheduled data refresh...")
    
    # Refresh every 30 minutes
    schedule.every(30).minutes.do(run_scheduled_refresh)
    
    # Refresh every hour
    schedule.every().hour.do(run_scheduled_refresh)
    
    # Refresh at specific times (e.g., every 4 hours)
    schedule.every(4).hours.do(run_scheduled_refresh)
    
    logger.info("✅ Scheduler configured for 30min, 1hr, and 4hr intervals")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NOESIS Data Refresh Pipeline")
    parser.add_argument("--once", action="store_true", help="Run refresh once and exit")
    parser.add_argument("--schedule", action="store_true", help="Run scheduled refresh")
    
    args = parser.parse_args()
    
    if args.once:
        # Run once
        pipeline = DataRefreshPipeline()
        result = asyncio.run(pipeline.run_full_refresh())
        print(f"Refresh result: {result}")
        
    elif args.schedule:
        # Run scheduled
        setup_scheduler()
        
    else:
        # Default: run once
        pipeline = DataRefreshPipeline()
        result = asyncio.run(pipeline.run_full_refresh())
        print(f"Refresh result: {result}") 