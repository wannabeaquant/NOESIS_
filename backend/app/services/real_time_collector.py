# NOTE: If you see 'Import "aiohttp" could not be resolved', select the correct Python interpreter (venv310) in VS Code.
#!/usr/bin/env python3
"""
Real-time Data Collector for NOESIS
Integrates with actual APIs for live data collection
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
from dataclasses import dataclass
import logging

# Try to import aiohttp, fallback to requests if not available
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    try:
        import requests
    except ImportError:
        requests = None
    HAS_AIOHTTP = False

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

class RealTimeCollector:
    def __init__(self):
        # API Keys (should be in environment variables)
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.weather_api_key = os.getenv('WEATHER_API_KEY')
        self.google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        
        # Protest-related keywords for monitoring
        self.protest_keywords = [
            "protest", "demonstration", "rally", "march", "gathering",
            "unrest", "disruption", "blockade", "occupation", "strike",
            "civil disobedience", "sit-in", "walkout", "boycott"
        ]
        
        # Location keywords for geolocation
        self.location_keywords = [
            "downtown", "city hall", "university", "campus", "square",
            "park", "street", "intersection", "building", "center"
        ]

    async def collect_twitter_data(self, location: Optional[str] = None) -> List[DataPoint]:
        """Collect real-time Twitter data using Twitter API v2"""
        if not self.twitter_bearer_token:
            logger.warning("Twitter API key not configured")
            return []
        
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
                    "max_results": 100,
                    "tweet.fields": "created_at,author_id,geo,public_metrics",
                    "user.fields": "location"
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        tweets = data.get('data', [])
                        
                        data_points = []
                        for tweet in tweets:
                            # Basic sentiment analysis (in real implementation, use proper ML model)
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
                                    "author_id": tweet['author_id'],
                                    "retweet_count": tweet['public_metrics']['retweet_count'],
                                    "like_count": tweet['public_metrics']['like_count']
                                }
                            ))
                        
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
            logger.warning("News API key not configured")
            return []
        
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
                    "pageSize": 50
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('articles', [])
                        
                        data_points = []
                        for article in articles:
                            sentiment_score = self._analyze_sentiment(article['title'] + " " + article.get('description', ''))
                            
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
                        
                        return data_points
                    else:
                        logger.error(f"News API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
            return []

    async def collect_reddit_data(self, subreddits: Optional[List[str]] = None) -> List[DataPoint]:
        """Collect Reddit data (using Reddit API or web scraping)"""
        if not subreddits:
            subreddits = ['protests', 'news', 'worldnews', 'politics']
        
        try:
            async with aiohttp.ClientSession() as session:
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
                            
                            for post in posts[:10]:  # Top 10 posts
                                post_data = post['data']
                                title = post_data['title']
                                
                                # Check if post is protest-related
                                if any(keyword.lower() in title.lower() for keyword in self.protest_keywords):
                                    sentiment_score = self._analyze_sentiment(title)
                                    
                                    data_points.append(DataPoint(
                                        source="reddit",
                                        content=title,
                                        location=None,  # Reddit doesn't provide location
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
                
                return data_points
                
        except Exception as e:
            logger.error(f"Error collecting Reddit data: {e}")
            return []

    async def collect_weather_data(self, location: str) -> Optional[DataPoint]:
        """Collect weather data that might affect protest likelihood"""
        if not self.weather_api_key:
            return None
        
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
        """Basic sentiment analysis (replace with proper ML model)"""
        # This is a simple rule-based sentiment analysis
        # In production, use a proper ML model like BERT or VADER
        
        negative_words = ['protest', 'demonstration', 'unrest', 'disruption', 'violence', 'arrest', 'conflict']
        positive_words = ['peaceful', 'peace', 'unity', 'solidarity', 'justice']
        
        text_lower = text.lower()
        
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
        # Bad weather (rain, snow, extreme cold) reduces protest likelihood
        # Good weather (sunny, mild) increases protest likelihood
        
        weather_condition = weather_data['weather'][0]['main'].lower()
        temperature = weather_data['main']['temp']
        
        if weather_condition in ['rain', 'snow', 'thunderstorm']:
            return -0.3  # Bad weather reduces protests
        elif weather_condition in ['clear', 'clouds'] and 15 <= temperature <= 25:
            return 0.2  # Good weather increases protests
        else:
            return 0.0  # Neutral

    async def collect_all_data(self, locations: Optional[List[str]] = None) -> List[DataPoint]:
        """Collect data from all sources"""
        if not locations:
            locations = ["New York", "Los Angeles", "Chicago", "Washington DC"]
        
        all_data = []
        
        # Collect from all sources concurrently
        tasks = []
        
        # Twitter data
        for location in locations:
            tasks.append(self.collect_twitter_data(location))
        
        # News data
        for location in locations:
            tasks.append(self.collect_news_data(location))
        
        # Reddit data
        tasks.append(self.collect_reddit_data())
        
        # Weather data
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
        
        return all_data

# Example usage
async def main():
    collector = RealTimeCollector()
    data = await collector.collect_all_data()
    print(f"Collected {len(data)} data points")
    
    for point in data[:5]:  # Show first 5
        print(f"{point.source}: {point.content[:50]}... (sentiment: {point.sentiment_score:.2f})")

if __name__ == "__main__":
    asyncio.run(main()) 