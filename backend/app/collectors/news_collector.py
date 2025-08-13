import requests
import feedparser
from typing import List, Dict
import os
from datetime import datetime
import json

class NewsCollector:
    def __init__(self, gnews_api_key: str = None):
        self.gnews_api_key = gnews_api_key or os.getenv("GNEWS_API_KEY")
        self.base_url = "https://gnews.io/api/v4/search"
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

    def collect_gnews(self) -> List[Dict]:
        """Collect news from GNews API"""
        posts = []
        
        if not self.gnews_api_key:
            print("GNews API key not configured. Using mock data.")
            return self._get_mock_data()
        
        try:
            kw = self.keywords
            # Build focused queries for protest-related content only
            protest_terms = kw["protest_unrest"][:5]  # Limit to most relevant terms
            escalation_terms = kw["escalation_violence"][:3]
            
            # Create targeted queries
            queries = []
            queries.extend(protest_terms)
            
            # Add high-relevance combinations
            for p in protest_terms[:3]:
                for e in escalation_terms[:2]:
                    queries.append(f'"{p}" AND "{e}"')  # Use exact phrase matching
            
            # Add location-specific queries for major cities prone to unrest
            major_cities = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore", "New York", "Los Angeles", "Chicago", "London", "Paris"]
            for city in major_cities[:5]:
                queries.append(f"protest {city}")
                queries.append(f"unrest {city}")
            
            queries = queries[:15]  # Limit total queries
            
            for keyword in queries:
                for country in ["us", "in"]:  # Query both US and India
                    try:
                        url = self.base_url
                        params = {
                            "q": keyword,
                            "token": self.gnews_api_key,
                            "lang": "en",
                            "country": country,
                            "max": 5,  # Reduced from 10 to focus on quality
                            "sortby": "publishedAt"
                        }
                        response = requests.get(url, params=params, timeout=10)
                        if response.status_code == 401:
                            print("GNews API key is invalid. Using mock data.")
                            return self._get_mock_data()
                        response.raise_for_status()
                        data = response.json()
                        if "articles" in data:
                            for article in data["articles"]:
                                try:
                                    title = self._sanitize_text(article.get("title", ""))
                                    description = self._sanitize_text(article.get("description", ""))
                                    content = title + " " + description
                                    
                                    if not content:
                                        continue
                                        
                                    # Enhanced filtering - check if content is actually protest-related
                                    if not self._is_protest_related(content):
                                        continue
                                    
                                    post = {
                                        "platform": "gnews",
                                        "content": content,
                                        "author": self._sanitize_text(article.get("source", {}).get("name", "Unknown")),
                                        "timestamp": article.get("publishedAt", ""),
                                        "location_raw": "",
                                        "link": article.get("url", ""),
                                        "extra": {
                                            "title": title,
                                            "source": article.get("source", {}).get("name", ""),
                                            "publishedAt": article.get("publishedAt", "")
                                        }
                                    }
                                    posts.append(post)
                                    print(f"Collected GNews article: {title[:100]}...")
                                except Exception as e:
                                    print(f"Error processing GNews article: {e}")
                                    continue
                    except Exception as e:
                        print(f"Error collecting GNews data for keyword {keyword} in country {country}: {e}")
                        continue
            
            print(f"Successfully collected {len(posts)} GNews articles")
            
        except Exception as e:
            print(f"Error collecting GNews data: {e}")
            print("Using mock data instead")
            posts = self._get_mock_data()
        
        return posts

    def collect_rss(self) -> List[Dict]:
        """Collect news from RSS feeds"""
        posts = []
        
        try:
            # RSS feeds for news sources
            rss_feeds = [
                "https://feeds.bbci.co.uk/news/rss.xml",
                "https://rss.cnn.com/rss/edition.rss",
                "https://feeds.reuters.com/Reuters/worldNews",
                "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
                "https://indianexpress.com/section/india/feed/",
                "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml"
            ]
            
            for feed_url in rss_feeds:
                try:
                    response = requests.get(feed_url, timeout=10)
                    response.raise_for_status()
                    
                    # Parse RSS feed
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(response.content)
                    
                    # Find items
                    items = root.findall(".//item")
                    
                    for item in items[:10]:  # Limit to 10 items per feed
                        try:
                            title_elem = item.find("title")
                            description_elem = item.find("description")
                            link_elem = item.find("link")
                            pub_date_elem = item.find("pubDate")
                            
                            if title_elem is not None:
                                title = self._sanitize_text(title_elem.text or "")
                                description = self._sanitize_text(description_elem.text or "") if description_elem is not None else ""
                                content = title + " " + description
                                
                                if not content:
                                    continue
                                # Enhanced filtering - check if content contains protest-related keywords and is actually relevant
                                if not self._is_protest_related(content):
                                    continue
                                    
                                post = {
                                    "platform": "rss",
                                    "content": content,
                                    "author": "RSS Feed",
                                    "timestamp": pub_date_elem.text if pub_date_elem is not None else "",
                                    "location_raw": "",
                                    "link": link_elem.text if link_elem is not None else "",
                                    "extra": {
                                        "title": title,
                                        "source": feed_url,
                                        "feed_type": "rss"
                                    }
                                }
                                posts.append(post)
                                print(f"Collected RSS article: {title[:100]}...")
                                
                        except Exception as e:
                            print(f"Error processing RSS item: {e}")
                            continue
                            
                except Exception as e:
                    print(f"Error collecting RSS from {feed_url}: {e}")
                    continue
            
            print(f"Successfully collected {len(posts)} RSS articles")
            
        except Exception as e:
            print(f"Error collecting RSS data: {e}")
            print("Using mock data instead")
            posts = self._get_mock_data()
        
        return posts

    def collect(self) -> List[Dict]:
        """Collect news from all sources (GNews + RSS)"""
        posts = []
        try:
            posts.extend(self.collect_gnews())
        except Exception as e:
            print(f"Error in collect_gnews: {e}")
        try:
            posts.extend(self.collect_rss())
        except Exception as e:
            print(f"Error in collect_rss: {e}")
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
    
    def _is_protest_related(self, content: str) -> bool:
        """Check if content is actually related to protests/civil unrest"""
        content_lower = content.lower()
        
        # Immediate exclusions for clearly irrelevant content
        exclude_terms = [
            "sports", "football", "basketball", "cricket", "tennis", "golf", "game", "match", "score",
            "movie", "film", "actor", "actress", "celebrity", "music", "concert", "album", "song",
            "stock", "trading", "investment", "crypto", "bitcoin", "market", "earnings", "profit",
            "weather", "rain", "storm", "hurricane", "flood", "earthquake", "snowfall",
            "recipe", "cooking", "food", "restaurant", "diet", "health", "fitness", "beauty",
            "fashion", "travel", "vacation", "tourism", "shopping", "sale", "discount"
        ]
        
        # If content has too many exclusion terms, reject it
        exclude_count = sum(1 for term in exclude_terms if term in content_lower)
        if exclude_count >= 2:
            return False
        
        # Positive protest indicators
        protest_indicators = [
            "protest", "demonstration", "rally", "march", "strike", "riot", "unrest", "uprising",
            "civil unrest", "police", "arrest", "clash", "confrontation", "tear gas", "crowd",
            "activists", "protesters", "demonstration", "blockade", "occupation"
        ]
        
        # Content must have at least one strong protest indicator
        indicator_count = sum(1 for term in protest_indicators if term in content_lower)
        
        return indicator_count >= 1

    def _get_mock_data(self) -> List[Dict]:
        """Generate mock news data for testing"""
        from datetime import datetime, timedelta
        
        mock_posts = [
            {
                "platform": "gnews",
                "content": "Breaking: Civil unrest reported in multiple cities. Authorities monitoring situation.",
                "author": "News Agency",
                "timestamp": (datetime.utcnow() - timedelta(hours=3)).isoformat(),
                "location_raw": "Multiple Cities",
                "link": "https://news.example.com/article/123",
                "extra": {"title": "Civil Unrest Report", "source": "News Agency"}
            },
            {
                "platform": "rss",
                "content": "Protest organizers announce new rally location. Police monitoring situation.",
                "author": "RSS Feed",
                "timestamp": (datetime.utcnow() - timedelta(hours=4)).isoformat(),
                "location_raw": "Downtown",
                "link": "https://rss.example.com/article/456",
                "extra": {"title": "Protest Rally Update", "source": "RSS Feed"}
            }
        ]
        
        print("Using mock news data")
        return mock_posts 