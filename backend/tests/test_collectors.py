#!/usr/bin/env python3
"""
Test script to check each collector individually
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Add the app directory to the path
sys.path.append('app')

from concurrent.futures import ThreadPoolExecutor, as_completed

def test_twitter_collector():
    """Test Twitter collector"""
    print("\n" + "="*50)
    print("TESTING TWITTER COLLECTOR")
    print("="*50)
    
    try:
        from app.collectors.twitter_collector import TwitterCollector
        
        collector = TwitterCollector()
        if not collector.bearer_token:
            print("Twitter API Key configured: False (using mock data)")
        else:
            print("Twitter API Key configured: True (using real data)")
        
        posts = collector.collect()
        if posts and posts[0].get('platform') == 'twitter' and 'mock' in posts[0].get('content', '').lower():
            print("TwitterCollector: Using mock data.")
        else:
            print(f"Twitter posts collected: {len(posts)}")
        
        if posts:
            print("Sample Twitter post:")
            print(f"  Platform: {posts[0].get('platform')}")
            print(f"  Content: {posts[0].get('content', '')[:100]}...")
            print(f"  Author: {posts[0].get('author')}")
            print(f"  Link: {posts[0].get('link')}")
        
        return posts
        
    except Exception as e:
        print(f"ERROR in Twitter collector: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_reddit_collector():
    """Test Reddit collector"""
    print("\n" + "="*50)
    print("TESTING REDDIT COLLECTOR")
    print("="*50)
    
    try:
        from app.collectors.reddit_collector import RedditCollector
        
        collector = RedditCollector()
        if not (collector.client_id and collector.client_secret and collector.user_agent):
            print("Reddit API configured: False (missing credentials, using mock data)")
        elif collector.reddit is None:
            print("Reddit API configured: False (initialization failed, using mock data)")
        else:
            # Try a simple API call to check if credentials work
            try:
                user = collector.reddit.user.me()
                print(f"Reddit API configured: True (Authenticated as: {user})")
            except Exception as e:
                print(f"Reddit API configured: False (Auth error: {e}) (using mock data)")
        
        posts = collector.collect()
        if posts and posts[0].get('platform') == 'reddit' and 'mock' in posts[0].get('content', '').lower():
            print("RedditCollector: Using mock data.")
        else:
            print(f"Reddit posts collected: {len(posts)}")
        
        if posts:
            print("Sample Reddit post:")
            print(f"  Platform: {posts[0].get('platform')}")
            print(f"  Content: {posts[0].get('content', '')[:100]}...")
            print(f"  Author: {posts[0].get('author')}")
            print(f"  Link: {posts[0].get('link')}")
        
        return posts
        
    except Exception as e:
        print(f"ERROR in Reddit collector: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_news_collector():
    """Test News collector"""
    print("\n" + "="*50)
    print("TESTING NEWS COLLECTOR")
    print("="*50)
    
    try:
        from app.collectors.news_collector import NewsCollector
        
        collector = NewsCollector()
        if not collector.gnews_api_key:
            print("GNews API Key configured: False (using mock data)")
        else:
            print("GNews API Key configured: True (using real data)")
        
        # Test GNews
        print("\n--- Testing GNews ---")
        gnews_posts = collector.collect_gnews()
        if gnews_posts and gnews_posts[0].get('platform') == 'gnews' and 'mock' in gnews_posts[0].get('content', '').lower():
            print("NewsCollector (GNews): Using mock data.")
        else:
            print(f"GNews articles collected: {len(gnews_posts)}")
        
        if gnews_posts:
            print("Sample GNews article:")
            print(f"  Platform: {gnews_posts[0].get('platform')}")
            print(f"  Content: {gnews_posts[0].get('content', '')[:100]}...")
            print(f"  Author: {gnews_posts[0].get('author')}")
            print(f"  Link: {gnews_posts[0].get('link')}")
        
        # Test RSS
        print("\n--- Testing RSS ---")
        rss_posts = collector.collect_rss()
        if rss_posts and rss_posts[0].get('platform') == 'rss' and 'mock' in rss_posts[0].get('content', '').lower():
            print("NewsCollector (RSS): Using mock data.")
        else:
            print(f"RSS articles collected: {len(rss_posts)}")
        
        if rss_posts:
            print("Sample RSS article:")
            print(f"  Platform: {rss_posts[0].get('platform')}")
            print(f"  Content: {rss_posts[0].get('content', '')[:100]}...")
            print(f"  Author: {rss_posts[0].get('author')}")
            print(f"  Link: {rss_posts[0].get('link')}")
        
        return gnews_posts + rss_posts
        
    except Exception as e:
        print(f"ERROR in News collector: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_data_orchestrator():
    """Test the data orchestrator's collect_all_data method"""
    print("\n" + "="*50)
    print("TESTING DATA ORCHESTRATOR")
    print("="*50)
    
    try:
        from app.services.data_orchestrator import DataOrchestrator
        from app.utils.database import get_db
        
        # Get a database session
        db = next(get_db())
        
        orchestrator = DataOrchestrator(db)
        
        print("Checking API keys...")
        has_keys = orchestrator._has_api_keys()
        print(f"Has API keys: {has_keys}")
        
        print("\nCollecting all data...")
        all_posts = orchestrator.collect_all_data()
        print(f"Total posts collected: {len(all_posts)}")
        
        if all_posts:
            print("Sample post from orchestrator:")
            print(f"  Platform: {all_posts[0].get('platform')}")
            print(f"  Content: {all_posts[0].get('content', '')[:100]}...")
            print(f"  Author: {all_posts[0].get('author')}")
        
        return all_posts
        
    except Exception as e:
        print(f"ERROR in Data Orchestrator: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_full_pipeline():
    """Test the full pipeline up to storing raw posts"""
    print("\n" + "="*50)
    print("TESTING FULL PIPELINE (up to storing raw posts)")
    print("="*50)
    
    try:
        from app.services.data_orchestrator import DataOrchestrator
        from app.utils.database import get_db
        
        # Get a database session
        db = next(get_db())
        
        orchestrator = DataOrchestrator(db)
        
        print("Step 1: Collecting raw data...")
        raw_posts = orchestrator.collect_all_data()
        print(f"Raw posts collected: {len(raw_posts)}")
        
        if not raw_posts:
            print("No raw posts collected, stopping here")
            return
        
        print("\nStep 2: Storing raw posts...")
        stored_posts = orchestrator.store_raw_posts(raw_posts)
        print(f"Raw posts stored: {len(stored_posts)}")
        
        if stored_posts:
            print("Sample stored post:")
            print(f"  ID: {stored_posts[0].id}")
            print(f"  Platform: {stored_posts[0].platform}")
            print(f"  Content: {stored_posts[0].content[:100]}...")
            print(f"  Author: {stored_posts[0].author}")
        
        return stored_posts
        
    except Exception as e:
        print(f"ERROR in full pipeline: {e}")
        import traceback
        traceback.print_exc()
        return []

def run_collectors_in_parallel():
    print("\n" + "="*50)
    print("RUNNING COLLECTORS IN PARALLEL")
    print("="*50)
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_name = {
            executor.submit(test_twitter_collector): 'twitter',
            executor.submit(test_reddit_collector): 'reddit',
            executor.submit(test_news_collector): 'news',
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                print(f"{name.capitalize()} collector generated an exception: {exc}")
                results[name] = []
    return results

def main():
    """Run all tests"""
    print("🚀 Testing NOESIS Collectors")
    print("="*60)
    
    # Run individual collectors in parallel
    parallel_results = run_collectors_in_parallel()
    twitter_posts = parallel_results.get('twitter', [])
    reddit_posts = parallel_results.get('reddit', [])
    news_posts = parallel_results.get('news', [])
    
    # Test data orchestrator (sequential)
    orchestrator_posts = test_data_orchestrator()
    
    # Test full pipeline (sequential)
    stored_posts = test_full_pipeline()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Twitter posts: {len(twitter_posts)}")
    print(f"Reddit posts: {len(reddit_posts)}")
    print(f"News posts: {len(news_posts)}")
    print(f"Orchestrator posts: {len(orchestrator_posts)}")
    print(f"Stored posts: {len(stored_posts) if stored_posts else 0}")
    
    if stored_posts:
        print("\n✅ All collectors working! The issue is likely in the NLP pipeline or incident creation.")
    else:
        print("\n❌ Issue found in data collection or storage phase.")

if __name__ == "__main__":
    main() 