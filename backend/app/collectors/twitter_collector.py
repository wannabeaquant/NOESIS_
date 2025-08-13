import tweepy
from typing import List, Dict
import os
from datetime import datetime, timedelta
import json

class TwitterCollector:
    def __init__(self, bearer_token: str = None):
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        if not self.bearer_token:
            print("Twitter API Bearer Token not configured. Real data will not be fetched.")
            self.client = None
        else:
            self.client = tweepy.Client(bearer_token=self.bearer_token, wait_on_rate_limit=True)

        # Load unrest keywords
        self.keywords = self._load_keywords()

    def _load_keywords(self):
        try:
            with open(os.path.join(os.path.dirname(__file__), '../../data/unrest_keywords.json'), 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading unrest_keywords.json: {e}")
            # Fallback to a basic set
            return {
                "protest_unrest": ["protest", "riot", "demonstration"],
                "escalation_violence": ["violence", "clash", "police"],
                "early_warning": ["gathering", "tension", "planned"],
                "triggers": ["election", "verdict", "policy"]
            }

    def collect(self) -> List[Dict]:
        """Collect recent tweets about protests and civil unrest"""
        posts = []
        
        if not self.client:
            print("Twitter API client not available, using mock data")
            return self._get_mock_data()
        
        try:
            # Replace the long search rule with a shorter one under 512 chars
            SEARCH_RULE = (
                '(protest OR unrest OR riot OR "civil unrest" OR strike OR march OR rally OR demonstration OR clash OR police OR arrest OR violence OR curfew OR lockdown OR military OR deployment OR crowd OR tension OR threat OR situation OR update) lang:en -is:retweet'
            )
            # Limit to avoid API abuse
            queries = [SEARCH_RULE]

            for query in queries:
                response = self.client.search_recent_tweets(
                    query=query,
                    max_results=25,
                    tweet_fields=["created_at", "author_id", "lang"],
                    expansions=["author_id"],
                    user_fields=["username", "location"]
                )
                tweets = response.data
                users = {u.id: u for u in response.includes["users"]} if response.includes and "users" in response.includes else {}
                if not tweets:
                    continue
                for tweet in tweets:
                    user = users.get(tweet.author_id, None)
                    content = self._sanitize_text(tweet.text)
                    if not content:
                        continue
                    post = {
                        "platform": "twitter",
                        "content": content,
                        "author": user.username if user else "unknown",
                        "timestamp": tweet.created_at.isoformat() if tweet.created_at else datetime.utcnow().isoformat(),
                        "location_raw": user.location if user and hasattr(user, 'location') and user.location else "",
                        "link": f"https://twitter.com/i/web/status/{tweet.id}",
                        "extra": {
                            "tweet_id": str(tweet.id),
                            "lang": tweet.lang or "en"
                        }
                    }
                    posts.append(post)
            print(f"Successfully collected {len(posts)} tweets")
        except Exception as e:
            print(f"Error collecting Twitter data: {e}")
            print("Using mock data instead")
            posts = self._get_mock_data()
        return posts

    def _sanitize_text(self, text: str) -> str:
        """Sanitize text to prevent encoding issues"""
        if not text:
            return ""
        
        try:
            # Handle bytes
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            
            # Handle None or non-string types
            if not isinstance(text, str):
                text = str(text)
            
            # Remove problematic characters
            import re
            # Remove control characters and non-printable characters
            text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
            # Remove characters that can't be encoded properly
            text = re.sub(r'[^\x20-\x7E\u00A0-\uFFFF]', ' ', text)
            # Replace multiple spaces and newlines
            text = re.sub(r'\s+', ' ', text)
            # Remove any remaining problematic characters
            text = ''.join(char for char in text if ord(char) < 65536)
            
            return text.strip()
        except Exception as e:
            print(f"Text sanitization error: {e}")
            return ""

    def _get_mock_data(self) -> List[Dict]:
        """Generate mock Twitter data for testing"""
        mock_posts = [
            {
                "platform": "twitter",
                "content": "Protest happening in downtown area. Police presence increased.",
                "author": "user123",
                "timestamp": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                "location_raw": "Downtown",
                "link": "https://twitter.com/user123/status/123456",
                "extra": {"tweet_id": "123456", "lang": "en"}
            },
            {
                "platform": "twitter",
                "content": "Large demonstration reported in city center. Multiple sources confirming.",
                "author": "news_reporter",
                "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                "location_raw": "City Center",
                "link": "https://twitter.com/news_reporter/status/123457",
                "extra": {"tweet_id": "123457", "lang": "en"}
            }
        ]
        
        print("Using mock Twitter data")
        return mock_posts 