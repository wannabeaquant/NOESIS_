#!/usr/bin/env python3
"""
Script to populate database with sample data for testing
"""
from app.utils.database import get_db
from app.models.incident import Incident
from app.models.raw_post import RawPost
from app.models.processed_post import ProcessedPost
from datetime import datetime, timedelta
import json

def create_sample_data():
    """Create sample incidents and posts for testing"""
    
    # Sample incidents
    sample_incidents = [
        {
            "title": "Protest at City Hall",
            "description": "Large demonstration against new policies",
            "location": "Downtown Area",
            "location_lat": 34.0522,
            "location_lng": -118.2437,
            "severity": "high",
            "status": "verified",
            "sources": ["twitter_123", "reddit_456", "news_789"]
        },
        {
            "title": "Student Demonstration at University",
            "description": "Peaceful protest for educational reforms",
            "location": "University Campus",
            "location_lat": 34.0689,
            "location_lng": -118.4452,
            "severity": "medium",
            "status": "verified",
            "sources": ["twitter_124", "reddit_457"]
        },
        {
            "title": "Workers Strike at Factory",
            "description": "Industrial workers demanding better conditions",
            "location": "Industrial District",
            "location_lat": 34.0625,
            "location_lng": -118.2386,
            "severity": "high",
            "status": "medium",
            "sources": ["news_790", "telegram_123"]
        },
        {
            "title": "Community Meeting Turned Tense",
            "description": "Local residents expressing concerns about development",
            "location": "Community Center",
            "location_lat": 34.0419,
            "location_lng": -118.2568,
            "severity": "low",
            "status": "unverified",
            "sources": ["twitter_125"]
        },
        {
            "title": "Traffic Protest Blocking Highway",
            "description": "Demonstrators blocking major highway intersection",
            "location": "Highway Intersection",
            "location_lat": 34.0736,
            "location_lng": -118.2400,
            "severity": "high",
            "status": "verified",
            "sources": ["twitter_126", "reddit_458", "news_791"]
        }
    ]
    
    # Sample raw posts
    sample_raw_posts = [
        {
            "platform": "twitter",
            "content": "Huge crowd gathering at City Hall! #protest #cityhall",
            "author": "user123",
            "timestamp": datetime.now() - timedelta(hours=2),
            "location_raw": "Downtown Area",
            "link": "https://twitter.com/user123/status/123456",
            "extra": {"followers": 1500, "verified": True}
        },
        {
            "platform": "reddit",
            "content": "Just saw police cars heading towards downtown. Something big happening.",
            "author": "reddit_user",
            "timestamp": datetime.now() - timedelta(hours=1, minutes=30),
            "location_raw": "Downtown Area",
            "link": "https://reddit.com/r/local_news/comments/456",
            "extra": {"upvotes": 45, "subreddit": "local_news"}
        },
        {
            "platform": "news",
            "content": "Breaking: Large demonstration reported at City Hall. Police on scene.",
            "author": "news_agency",
            "timestamp": datetime.now() - timedelta(hours=1),
            "location_raw": "Downtown Area",
            "link": "https://localnews.com/article/789",
            "extra": {"reliability": 0.9, "outlet": "Local News"}
        }
    ]
    
    # Sample processed posts
    sample_processed_posts = [
        {
            "raw_post_id": 1,
            "protest_score": 0.8,
            "sentiment_score": -0.7,
            "location_lat": 34.0522,
            "location_lng": -118.2437,
            "language": "en",
            "platform": "twitter",
            "link": "https://twitter.com/user123/status/123456",
            "entities": {"protest": 0.8, "crowd": 0.9, "city_hall": 0.9},
            "status": "verified"
        },
        {
            "raw_post_id": 2,
            "protest_score": 0.6,
            "sentiment_score": -0.3,
            "location_lat": 34.0522,
            "location_lng": -118.2437,
            "language": "en",
            "platform": "reddit",
            "link": "https://reddit.com/r/local_news/comments/456",
            "entities": {"police": 0.8, "downtown": 0.7, "activity": 0.6},
            "status": "medium"
        },
        {
            "raw_post_id": 3,
            "protest_score": 0.7,
            "sentiment_score": -0.6,
            "location_lat": 34.0522,
            "location_lng": -118.2437,
            "language": "en",
            "platform": "news",
            "link": "https://localnews.com/article/789",
            "entities": {"demonstration": 0.9, "police": 0.8, "city_hall": 0.9},
            "status": "verified"
        }
    ]
    
    return sample_incidents, sample_raw_posts, sample_processed_posts

def populate_database():
    """Populate the database with sample data"""
    print("Populating database with sample data...")
    
    try:
        # Get database session
        db = next(get_db())
        
        # Create sample data
        sample_incidents, sample_raw_posts, sample_processed_posts = create_sample_data()
        
        # Add raw posts
        print("Adding raw posts...")
        for post_data in sample_raw_posts:
            raw_post = RawPost(**post_data)
            db.add(raw_post)
        db.commit()
        print(f"✅ Added {len(sample_raw_posts)} raw posts")
        
        # Add processed posts
        print("Adding processed posts...")
        for post_data in sample_processed_posts:
            processed_post = ProcessedPost(**post_data)
            db.add(processed_post)
        db.commit()
        print(f"✅ Added {len(sample_processed_posts)} processed posts")
        
        # Add incidents
        print("Adding incidents...")
        for incident_data in sample_incidents:
            incident = Incident(**incident_data)
            db.add(incident)
        db.commit()
        print(f"✅ Added {len(sample_incidents)} incidents")
        
        print("🎉 Database populated successfully!")
        print(f"📊 Summary:")
        print(f"   - Raw posts: {len(sample_raw_posts)}")
        print(f"   - Processed posts: {len(sample_processed_posts)}")
        print(f"   - Incidents: {len(sample_incidents)}")
        
    except Exception as e:
        print(f"❌ Error populating database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    populate_database() 