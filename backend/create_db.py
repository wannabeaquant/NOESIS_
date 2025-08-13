#!/usr/bin/env python3
"""
Script to create database tables manually
"""
from app.utils.database import create_tables
from app.models.base import Base
from app.models.raw_post import RawPost
from app.models.processed_post import ProcessedPost
from app.models.incident import Incident
from app.models.alert_subscriber import AlertSubscriber

def main():
    print("Creating database tables...")
    try:
        create_tables()
        print("✅ Database tables created successfully!")
        print("Tables created:")
        print("- raw_posts")
        print("- processed_posts") 
        print("- incidents")
        print("- alert_subscribers")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")

if __name__ == "__main__":
    main() 