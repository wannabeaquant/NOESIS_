# NOTE: If you see 'Import "aiohttp" could not be resolved', select the correct Python interpreter (venv310) in VS Code.
#!/usr/bin/env python3
"""
Simplified Real-time Data Collector for NOESIS
Works with free APIs and includes rate limiting
"""

import asyncio
import aiohttp
import requests
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataPoint:
    source: str
    content: str
    location: Optional[str]
    timestamp: datetime
    sentiment_score: float
    confidence: float
    metadata: Dict[str, Any]

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window  # in seconds
        self.requests = []
    
    async def wait_if_needed(self):
        """Wait if we've hit the rate limit"""
        now = time.time()
        
        # Remove old requests outside the time window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        # If we've hit the limit, wait
        if len(self.requests) >= self.max_requests:
            oldest_request = min(self.requests)
            wait_time = self.time_window - (now - oldest_request) + 1
            logger.info(f"Rate limit hit, waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
            self.requests = []
        
        # Record this request
        self.requests.append(now)

class SimpleRealTimeCollector:
    def __init__(self):
        # API Keys from environment variables
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.weather_api_key = os.getenv('WEATHER_API_KEY')
        
        # Rate limiters (free tier limits)
        self.twitter_limiter = RateLimiter(max_requests=300, time_window=900)  # 300 per 15 min
        self.news_limiter = RateLimiter(max_requests=100, time_window=86400)   # 100 per day
        self.weather_limiter = RateLimiter(max_requests=60, time_window=60)    # 60 per min
        
        # Protest-related keywords
        self.protest_keywords = [
            "protest", "demonstration", "rally", "march", "gathering",
            "unrest", "disruption", "blockade", "occupation", "strike"
        ]

    async def collect_twitter_data(self, location: Optional[str] = None) -> List[DataPoint]:
        """Collect Twitter data (if API key is available)"""
        if not self.twitter_bearer_token:
            logger.info("Twitter API key not configured, skipping Twitter data")
            return []
        
        await self.twitter_limiter.wait_if_needed()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Search for protest-related tweets
                query = " OR ".join([f'"{keyword}"' for keyword in self.protest_keywords])
                if location:
                    query += f" AND {location}"
                
                url = "https://api.twitter.com/2/tweets/search/recent"
                headers = {
                    "Authorization": f"Bearer {self.twitter_bearer_token}",
                    "Content-Type": "application/json"
                }
                params = {
                    "query": query,
                    "max_results": 10,  # Reduced for free tier
                    "tweet.fields": "created_at,author_id,geo,public_metrics"
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        tweets = data.get('data', [])
                        
                        data_points = []
                        for tweet in tweets:
                            sentiment_score = self._analyze_sentiment(tweet['text'])
                            
                            data_points.append(DataPoint(
                                source="twitter",
                                content=tweet['text'],
                                location=tweet.get('geo', {}).get('place_id'),
                                timestamp=datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00')),
                                sentiment_score=sentiment_score,
                                confidence=0.8,
                                metadata={
                                    "tweet_id": tweet['id'],
                                    "retweet_count": tweet['public_metrics']['retweet_count'],
                                    "like_count": tweet['public_metrics']['like_count']
                                }
                            ))
                        
                        logger.info(f"Collected {len(data_points)} tweets")
                        return data_points
                    else:
                        logger.error(f"Twitter API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error collecting Twitter data: {e}")
            return []

    async def collect_news_data(self, location: Optional[str] = None) -> List[DataPoint]:
        """Collect news data using News API"""
        if not self.news_api_key:
            logger.info("News API key not configured, skipping news data")
            return []
        
        await self.news_limiter.wait_if_needed()
        
        try:
            async with aiohttp.ClientSession() as session:
                query = " OR ".join(self.protest_keywords)
                if location:
                    query += f" AND {location}"
                
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": query,
                    "apiKey": self.news_api_key,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 10  # Reduced for free tier
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('articles', [])
                        
                        data_points = []
                        for article in articles:
                            sentiment_score = self._analyze_sentiment(
                                article['title'] + " " + article.get('description', '')
                            )
                            
                            data_points.append(DataPoint(
                                source="news",
                                content=article['title'],
                                location=article.get('source', {}).get('name'),
                                timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                                sentiment_score=sentiment_score,
                                confidence=0.9,
                                metadata={
                                    "url": article['url'],
                                    "source": article['source']['name'],
                                    "author": article.get('author')
                                }
                            ))
                        
                        logger.info(f"Collected {len(data_points)} news articles")
                        return data_points
                    else:
                        logger.error(f"News API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
            return []

    async def collect_reddit_data(self) -> List[DataPoint]:
        """Collect Reddit data (no API key needed)"""
        try:
            async with aiohttp.ClientSession() as session:
                subreddits = ['news', 'worldnews', 'politics']
                data_points = []
                
                for subreddit in subreddits:
                    url = f"https://www.reddit.com/r/{subreddit}/hot.json"
                    headers = {
                        "User-Agent": "NOESIS-Intelligence-Platform/1.0"
                    }
                    
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            posts = data.get('data', {}).get('children', [])
                            
                            for post in posts[:5]:  # Top 5 posts
                                post_data = post['data']
                                title = post_data['title']
                                
                                # Check if post is protest-related
                                if any(keyword.lower() in title.lower() for keyword in self.protest_keywords):
                                    sentiment_score = self._analyze_sentiment(title)
                                    
                                    data_points.append(DataPoint(
                                        source="reddit",
                                        content=title,
                                        location=None,
                                        timestamp=datetime.fromtimestamp(post_data['created_utc']),
                                        sentiment_score=sentiment_score,
                                        confidence=0.7,
                                        metadata={
                                            "subreddit": subreddit,
                                            "score": post_data['score'],
                                            "num_comments": post_data['num_comments'],
                                            "url": post_data['url']
                                        }
                                    ))
                
                logger.info(f"Collected {len(data_points)} Reddit posts")
                return data_points
                
        except Exception as e:
            logger.error(f"Error collecting Reddit data: {e}")
            return []

    async def collect_weather_data(self, location: str = "New York") -> Optional[DataPoint]:
        """Collect weather data"""
        if not self.weather_api_key:
            logger.info("Weather API key not configured, skipping weather data")
            return None
        
        await self.weather_limiter.wait_if_needed()
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "http://api.openweathermap.org/data/2.5/weather"
                params = {
                    "q": location,
                    "appid": self.weather_api_key,
                    "units": "metric"
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Weather conditions that might affect protests
                        weather_score = self._calculate_weather_score(data)
                        
                        return DataPoint(
                            source="weather",
                            content=f"Weather in {location}: {data['weather'][0]['description']}",
                            location=location,
                            timestamp=datetime.utcnow(),
                            sentiment_score=weather_score,
                            confidence=0.95,
                            metadata={
                                "temperature": data['main']['temp'],
                                "humidity": data['main']['humidity'],
                                "wind_speed": data['wind']['speed'],
                                "weather_condition": data['weather'][0]['main']
                            }
                        )
                    else:
                        logger.error(f"Weather API error: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error collecting weather data: {e}")
            return None

    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis"""
        text_lower = text.lower()
        
        negative_words = ['protest', 'demonstration', 'unrest', 'disruption', 'violence', 'arrest', 'conflict']
        positive_words = ['peaceful', 'peace', 'unity', 'solidarity', 'justice']
        
        negative_count = sum(1 for word in negative_words if word in text_lower)
        positive_count = sum(1 for word in positive_words if word in text_lower)
        
        if negative_count > positive_count:
            return -0.5 - (negative_count * 0.1)
        elif positive_count > negative_count:
            return 0.3 + (positive_count * 0.1)
        else:
            return 0.0

    def _calculate_weather_score(self, weather_data: Dict) -> float:
        """Calculate how weather might affect protest likelihood"""
        weather_condition = weather_data['weather'][0]['main'].lower()
        temperature = weather_data['main']['temp']
        
        if weather_condition in ['rain', 'snow', 'thunderstorm']:
            return -0.3  # Bad weather reduces protests
        elif weather_condition in ['clear', 'clouds'] and 15 <= temperature <= 25:
            return 0.2  # Good weather increases protests
        else:
            return 0.0  # Neutral

    async def collect_all_data(self, locations: Optional[List[str]] = None) -> List[DataPoint]:
        """Collect data from all available sources"""
        if not locations:
            locations = ["New York", "Los Angeles", "Chicago"]
        
        all_data = []
        
        # Collect from all sources concurrently
        tasks = []
        
        # Twitter data (if API key available)
        if self.twitter_bearer_token:
            for location in locations:
                tasks.append(self.collect_twitter_data(location))
        
        # News data (if API key available)
        if self.news_api_key:
            for location in locations:
                tasks.append(self.collect_news_data(location))
        
        # Reddit data (always available)
        tasks.append(self.collect_reddit_data())
        
        # Weather data (if API key available)
        if self.weather_api_key:
            for location in locations:
                tasks.append(self.collect_weather_data(location))
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)
            elif isinstance(result, DataPoint):
                all_data.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Data collection error: {result}")
        
        logger.info(f"Total data points collected: {len(all_data)}")
        return all_data

# Test function
async def test_collector():
    """Test the data collector"""
    collector = SimpleRealTimeCollector()
    
    print("Testing data collection...")
    print(f"Twitter API configured: {bool(collector.twitter_bearer_token)}")
    print(f"News API configured: {bool(collector.news_api_key)}")
    print(f"Weather API configured: {bool(collector.weather_api_key)}")
    
    data = await collector.collect_all_data()
    print(f"\nCollected {len(data)} data points:")
    
    for point in data[:3]:  # Show first 3
        print(f"- {point.source}: {point.content[:50]}... (sentiment: {point.sentiment_score:.2f})")

if __name__ == "__main__":
    asyncio.run(test_collector()) 