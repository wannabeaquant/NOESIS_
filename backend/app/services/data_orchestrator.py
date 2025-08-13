from typing import List, Dict
from sqlalchemy.orm import Session
from app.collectors.twitter_collector import TwitterCollector
from app.collectors.reddit_collector import RedditCollector
from app.collectors.news_collector import NewsCollector
from app.collectors.satellite_collector import SatelliteCollector
from app.collectors.financial_collector import FinancialCollector
from app.collectors.iot_collector import IoTCollector
from app.services.nlp_pipeline import NLPPipeline
from app.services.verification import VerificationService
from app.models.raw_post import RawPost
from app.models.processed_post import ProcessedPost
from app.models.incident import Incident
from app.alerts.telegram_bot import TelegramAlertBot
from app.alerts.email_alert import EmailAlert
import json
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import os
import json

class DataOrchestrator:
    def __init__(self, db: Session):
        self.db = db
        self.twitter_collector = TwitterCollector()
        self.reddit_collector = RedditCollector()
        self.news_collector = NewsCollector()
        self.satellite_collector = SatelliteCollector()
        self.financial_collector = FinancialCollector()
        self.iot_collector = IoTCollector()
        self.nlp_pipeline = NLPPipeline()
        self.verification_service = VerificationService()
        self.telegram_bot = TelegramAlertBot()
        self.email_alert = EmailAlert()
        # Audit logging setup
        self.audit_log_threshold = float(os.getenv("NLP_FILTER_THRESHOLD", "0.2"))
        self.audit_logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'logs'))
        try:
            os.makedirs(self.audit_logs_dir, exist_ok=True)
        except Exception:
            pass
        self.audit_log_file = os.path.join(self.audit_logs_dir, 'noesis_pipeline_audit.log')
        # Create a dedicated logger for pipeline audit
        self.audit_logger = logging.getLogger("noesis_pipeline_audit")
        self.audit_logger.propagate = False  # Prevent duplicate logging
        if not self.audit_logger.handlers:
            self.audit_logger.setLevel(logging.INFO)
            try:
                file_handler = logging.FileHandler(self.audit_log_file, encoding='utf-8')
                formatter = logging.Formatter('%(message)s')  # Just the JSON message, no timestamp prefix
                file_handler.setFormatter(formatter)
                self.audit_logger.addHandler(file_handler)
            except Exception:
                # Fallback to console-only if file handler fails
                stream_handler = logging.StreamHandler()
                formatter = logging.Formatter('AUDIT: %(message)s')
                stream_handler.setFormatter(formatter)
                self.audit_logger.addHandler(stream_handler)

    def _write_audit_event(self, event: dict):
        """Write a single JSON line audit event without breaking pipeline."""
        try:
            event_with_time = {**event, "ts": datetime.utcnow().isoformat()}
            json_line = json.dumps(event_with_time, ensure_ascii=False)
            self.audit_logger.info(json_line)
        except Exception:
            # Never break the pipeline due to logging
            pass

    def run_collection_cycle(self):
        """Run a complete data collection and processing cycle in parallel for all collectors"""
        print("Starting data collection cycle (parallel)...")
        raw_posts = []
        errors = []
        collectors = [
            ("twitter", self.twitter_collector.collect),
            ("reddit", self.reddit_collector.collect),
            ("news", self.news_collector.collect),
            ("satellite", self.satellite_collector.collect),
            ("financial", self.financial_collector.collect),
            ("iot", self.iot_collector.collect)
        ]
        
        def run_collector(name, func):
            try:
                print(f"Running {name} collector...")
                posts = func()
                print(f"{name.title()} collector returned {len(posts)} posts.")
                return posts
            except Exception as e:
                print(f"Error in {name} collector: {e}")
                errors.append((name, str(e)))
                return []
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_name = {executor.submit(run_collector, name, func): name for name, func in collectors}
            for future in as_completed(future_to_name):
                posts = future.result()
                if posts:
                    raw_posts.extend(posts)
        
        print(f"Collected {len(raw_posts)} raw posts from all collectors.")
        if errors:
            print("Some collectors failed or were rate-limited:")
            for name, err in errors:
                print(f"  - {name}: {err}")
        
        # Store raw posts in database
        stored_posts = self.store_raw_posts(raw_posts)
        print(f"Stored {len(stored_posts)} raw posts")
        if stored_posts:
            print("Sample stored post:", stored_posts[0].__dict__)
        
        # Process posts through NLP pipeline
        processed_posts = self.process_posts(stored_posts)
        print(f"Processed {len(processed_posts)} posts")
        if processed_posts:
            print("Sample processed post:", processed_posts[0].__dict__)
        
        # Verify and create incidents
        incidents = self.create_incidents(processed_posts)
        print(f"Created {len(incidents)} incidents")
        if incidents:
            print("Sample incident:", incidents[0].__dict__)
        
        # Send alerts for new incidents
        self.send_alerts(incidents)
        print("Alerts sent")
        
        return {
            "raw_posts": len(raw_posts),
            "processed_posts": len(processed_posts),
            "incidents": len(incidents),
            "errors": errors
        }

    def collect_all_data(self) -> List[Dict]:
        """Collect data from all sources"""
        all_posts = []
        
        # For testing purposes, add some mock data if no API keys are configured
        if not self._has_api_keys():
            print("No API keys configured, using mock data for testing")
            all_posts = self._get_mock_data()
            return all_posts
        
        # Collect from Twitter
        try:
            twitter_posts = self.twitter_collector.collect()
            all_posts.extend(twitter_posts)
            print(f"Collected {len(twitter_posts)} Twitter posts")
        except Exception as e:
            print(f"Error collecting Twitter data: {e}")
        
        # Collect from Reddit
        try:
            reddit_posts = self.reddit_collector.collect()
            all_posts.extend(reddit_posts)
            print(f"Collected {len(reddit_posts)} Reddit posts")
        except Exception as e:
            print(f"Error collecting Reddit data: {e}")
        
        # Collect from News
        try:
            gnews_posts = self.news_collector.collect_gnews()
            all_posts.extend(gnews_posts)
            print(f"Collected {len(gnews_posts)} GNews articles")
            
            rss_posts = self.news_collector.collect_rss()
            all_posts.extend(rss_posts)
            print(f"Collected {len(rss_posts)} RSS articles")
        except Exception as e:
            print(f"Error collecting news data: {e}")
        
        return all_posts

    def _has_api_keys(self) -> bool:
        """Check if any API keys are configured"""
        import os
        return any([
            os.getenv("TWITTER_BEARER_TOKEN"),
            os.getenv("REDDIT_CLIENT_ID"),
            os.getenv("GNEWS_API_KEY")
        ])

    def _get_mock_data(self) -> List[Dict]:
        """Generate mock data for testing"""
        from datetime import datetime, timedelta
        
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
                "platform": "reddit",
                "content": "Large demonstration reported in city center. Multiple sources confirming.",
                "author": "reddit_user",
                "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                "location_raw": "City Center",
                "link": "https://reddit.com/r/news/comments/123456",
                "extra": {"subreddit": "news", "score": 150}
            },
            {
                "platform": "gnews",
                "content": "Breaking: Civil unrest reported in multiple cities. Authorities monitoring situation.",
                "author": "News Agency",
                "timestamp": (datetime.utcnow() - timedelta(hours=3)).isoformat(),
                "location_raw": "Multiple Cities",
                "link": "https://news.example.com/article/123",
                "extra": {"title": "Civil Unrest Report", "source": "News Agency"}
            }
        ]
        
        return mock_posts

    def store_raw_posts(self, raw_posts: List[Dict]) -> List[RawPost]:
        """Store raw posts in database"""
        stored_posts = []
        
        for post_data in raw_posts:
            try:
                # Check if post already exists (by link)
                link = post_data.get("link", "")
                if link:
                    existing = self.db.query(RawPost).filter(RawPost.link == link).first()
                    if existing:
                        # Log duplicate skip for completeness
                        try:
                            extra_src = None
                            extra = post_data.get("extra", {})
                            if isinstance(extra, dict):
                                extra_src = extra.get("source")
                            self._write_audit_event({
                                "stage": "SCRAPED_DUPLICATE_SKIPPED",
                                "raw_post_id": getattr(existing, 'id', None),
                                "platform": post_data.get("platform", "unknown"),
                                "source": extra_src,
                                "link": link,
                                "title": (extra.get("title") if isinstance(extra, dict) else None)
                            })
                        except Exception:
                            pass
                        continue
                
                # Clean and sanitize data with better error handling
                try:
                    content = self.sanitize_text(post_data.get("content", ""))
                except Exception as e:
                    print(f"Error sanitizing content: {e}")
                    content = "content"
                
                try:
                    author = self.sanitize_text(post_data.get("author", ""))
                except Exception as e:
                    print(f"Error sanitizing author: {e}")
                    author = "author"
                
                try:
                    location_raw = self.sanitize_text(post_data.get("location_raw", ""))
                except Exception as e:
                    print(f"Error sanitizing location: {e}")
                    location_raw = ""
                
                # Handle timestamp safely
                timestamp_str = post_data.get("timestamp", "")
                try:
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    else:
                        timestamp = datetime.utcnow()
                except:
                    timestamp = datetime.utcnow()
                
                # Sanitize extra data
                extra_data = post_data.get("extra", {})
                try:
                    # Convert extra data to string-safe format
                    if isinstance(extra_data, dict):
                        sanitized_extra = {}
                        for key, value in extra_data.items():
                            if isinstance(value, str):
                                sanitized_extra[str(key)] = self.sanitize_text(value)
                            else:
                                sanitized_extra[str(key)] = str(value)
                        extra_data = sanitized_extra
                    else:
                        extra_data = {"data": str(extra_data)}
                except Exception as e:
                    print(f"Error sanitizing extra data: {e}")
                    extra_data = {"error": "data_sanitization_failed"}
                
                raw_post = RawPost(
                    platform=post_data.get("platform", "unknown"),
                    content=content,
                    author=author,
                    timestamp=timestamp,
                    location_raw=location_raw,
                    link=link,
                    extra=extra_data
                )
                
                self.db.add(raw_post)
                # Flush to populate primary key for audit logging
                try:
                    self.db.flush()
                except Exception:
                    pass
                stored_posts.append(raw_post)
                # Audit log the scraped raw post
                try:
                    title = None
                    headline = None
                    if isinstance(extra_data, dict):
                        title = extra_data.get("title")
                        headline = extra_data.get("headline")
                    self._write_audit_event({
                        "stage": "SCRAPED",
                        "raw_post_id": getattr(raw_post, 'id', None),
                        "platform": raw_post.platform,
                        "source": (extra_data.get("source") if isinstance(extra_data, dict) else None),
                        "link": raw_post.link,
                        "title": title,
                        "headline": headline,
                        "content": content
                    })
                except Exception:
                    pass
                
            except Exception as e:
                print(f"Error storing raw post: {e}")
                import traceback
                print(traceback.format_exc())
                continue
        
        try:
            self.db.commit()
        except Exception as e:
            print(f"Error committing raw posts: {e}")
            self.db.rollback()
            
        return stored_posts

    def sanitize_text(self, text: str) -> str:
        """Sanitize text to handle encoding issues"""
        if not text:
            return ""
        
        try:
            # Handle bytes
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            
            # Handle None or non-string types
            if not isinstance(text, str):
                text = str(text)
            
            # Remove problematic characters more aggressively
            import re
            # Remove non-printable characters and control characters
            text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
            # Remove characters that can't be encoded in SQLite
            text = re.sub(r'[^\x20-\x7E\u00A0-\uFFFF]', ' ', text)
            # Replace multiple spaces and newlines
            text = re.sub(r'\s+', ' ', text)
            # Remove any remaining problematic characters
            text = ''.join(char for char in text if ord(char) < 65536)
            
            return text.strip()
        except Exception as e:
            print(f"Text sanitization error: {e}")
            return "text"

    def process_posts(self, raw_posts: List[RawPost]) -> List[ProcessedPost]:
        """Process raw posts through NLP pipeline"""
        processed_posts = []
        
        for raw_post in raw_posts:
            try:
                # Convert to dict for NLP processing with safe data handling
                post_dict = {
                    "id": raw_post.id,
                    "content": raw_post.content or "",
                    "platform": raw_post.platform or "",
                    "link": raw_post.link or "",
                    "location_raw": raw_post.location_raw or "",
                    "timestamp": raw_post.timestamp.isoformat() if raw_post.timestamp is not None else "",
                    "title": (raw_post.extra.get("title") if raw_post.extra else None),
                    "headline": (raw_post.extra.get("headline") if raw_post.extra else None)
                }
                
                # Ensure all string fields are properly sanitized
                for key, value in post_dict.items():
                    if isinstance(value, str):
                        post_dict[key] = self.sanitize_text(value)
                
                # Process through NLP pipeline with error handling
                try:
                    processed_data = self.nlp_pipeline.process(post_dict)
                except Exception as e:
                    print(f"Error in NLP pipeline for post {raw_post.id}: {e}")
                    # Create fallback processed data
                    processed_data = {
                        "protest_score": 0.0,
                        "sentiment_score": 0.0,
                        "location_lat": None,
                        "location_lng": None,
                        "language": "en",
                        "platform": post_dict.get("platform", "unknown"),
                        "link": post_dict.get("link", ""),
                        "entities": {},
                        "status": "unverified",
                        "classification_info": {"decision": "error_in_pipeline", "details": {"error": str(e)}}
                    }
                
                # Audit log: if NLP marks it below threshold, record as filtered
                try:
                    protest_score = float(processed_data.get("protest_score", 0.0))
                    classification_info = processed_data.get("classification_info", {}) or {}
                    if protest_score < self.audit_log_threshold:
                        # Gather optional title/headline from RawPost.extra
                        extra = raw_post.extra if isinstance(raw_post.extra, dict) else {}
                        self._write_audit_event({
                            "stage": "FILTERED_BY_NLP",
                            "raw_post_id": raw_post.id,
                            "platform": raw_post.platform,
                            "link": raw_post.link,
                            "source": extra.get("source"),
                            "title": extra.get("title"),
                            "headline": extra.get("headline"),
                            "content": raw_post.content,
                            "protest_score": protest_score,
                            "threshold": self.audit_log_threshold,
                            "classification_info": classification_info
                        })
                except Exception:
                    pass

                # Store processed post with safe data handling
                try:
                    processed_post = ProcessedPost(
                        raw_post_id=raw_post.id,
                        protest_score=float(processed_data.get("protest_score", 0.0)),
                        sentiment_score=float(processed_data.get("sentiment_score", 0.0)),
                        location_lat=processed_data.get("location_lat"),
                        location_lng=processed_data.get("location_lng"),
                        language=str(processed_data.get("language", "en")),
                        platform=str(processed_data.get("platform", "unknown")),
                        link=str(processed_data.get("link", "")),
                        entities=processed_data.get("entities", {}),
                        status=str(processed_data.get("status", "unverified"))
                    )
                    
                    self.db.add(processed_post)
                    processed_posts.append(processed_post)
                    
                except Exception as e:
                    print(f"Error creating processed post for {raw_post.id}: {e}")
                    # Create a minimal processed post with default values
                    try:
                        processed_post = ProcessedPost(
                            raw_post_id=raw_post.id,
                            protest_score=0.0,
                            sentiment_score=0.0,
                            location_lat=None,
                            location_lng=None,
                            language="en",
                            platform="unknown",
                            link="",
                            entities={},
                            status="unverified"
                        )
                        self.db.add(processed_post)
                        processed_posts.append(processed_post)
                    except Exception as e2:
                        print(f"Error creating fallback processed post: {e2}")
                
            except Exception as e:
                print(f"Error processing post {raw_post.id}: {e}")
                import traceback
                print(traceback.format_exc())
                # Create a minimal processed post with default values
                try:
                    processed_post = ProcessedPost(
                        raw_post_id=raw_post.id,
                        protest_score=0.0,
                        sentiment_score=0.0,
                        location_lat=None,
                        location_lng=None,
                        language="en",
                        platform="unknown",
                        link="",
                        entities={},
                        status="unverified"
                    )
                    self.db.add(processed_post)
                    processed_posts.append(processed_post)
                except Exception as e2:
                    print(f"Error creating fallback processed post: {e2}")
        
        try:
            self.db.commit()
        except Exception as e:
            print(f"Error committing processed posts: {e}")
            self.db.rollback()
            
        return processed_posts

    def create_incidents(self, processed_posts: List[ProcessedPost]) -> List[Incident]:
        """Create incidents from processed posts"""
        # Build a mapping from raw_post_id to RawPost
        raw_post_map = {raw_post.id: raw_post for raw_post in self.db.query(RawPost).all()}
        
        # Convert to dict for verification service
        posts_dict = []
        for post in processed_posts:
            raw_post = raw_post_map.get(post.raw_post_id)
            if raw_post:
                title = raw_post.extra.get("title") if raw_post.extra else None
                headline = raw_post.extra.get("headline") if raw_post.extra else None
                content = raw_post.content
            else:
                title = None
                headline = None
                content = None
            posts_dict.append({
                "id": post.id,
                "protest_score": post.protest_score,
                "sentiment_score": post.sentiment_score,
                "location_lat": post.location_lat,
                "location_lng": post.location_lng,
                "platform": post.platform,
                "link": post.link,
                "timestamp": post.timestamp.isoformat(),
                "title": title,
                "headline": headline,
                "content": content
            })
        
        print(f"[DEBUG] create_incidents: {len(posts_dict)} processed posts passed to verification.")
        # Verify and create incidents
        incident_data = self.verification_service.verify(posts_dict)
        print(f"[DEBUG] create_incidents: {len(incident_data)} incidents returned from verification.")
        if incident_data:
            print(f"[DEBUG] First incident: {incident_data[0]}")
        
        incidents = []
        for data in incident_data:
            incident = Incident(
                title=data.get("title"),
                description=data.get("description"),
                sources=data.get("sources", []),
                location=data.get("location"),
                location_lat=data.get("location_lat"),
                location_lng=data.get("location_lng"),
                severity=data.get("severity"),
                status=data.get("status")
            )
            self.db.add(incident)
            incidents.append(incident)
        self.db.commit()
        print(f"[DEBUG] create_incidents: {len(incidents)} incidents committed to DB.")
        return incidents

    def send_alerts(self, incidents: List[Incident]):
        """Send alerts for new incidents"""
        for incident in incidents:
            if incident.status in ["verified", "medium"] and incident.severity in ["medium", "high"]:
                # Convert to dict for alert services
                incident_dict = {
                    "title": incident.title,
                    "description": incident.description,
                    "location": incident.location,
                    "severity": incident.severity,
                    "status": incident.status,
                    "sources": incident.sources
                }
                
                # Send Telegram alert
                self.telegram_bot.broadcast_incident(incident_dict)
                
                # Send email alert (to admin)
                # self.email_alert.send_alert(
                #     to_email="admin@noesis.com",
                #     subject=f"NOESIS Alert: {incident.severity.title()} severity incident",
                #     message=f"New incident detected: {incident.title}"
                # ) 