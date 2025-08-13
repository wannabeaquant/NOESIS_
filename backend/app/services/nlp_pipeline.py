from typing import Dict, List
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import geotext
import requests
import os
from langdetect import detect, DetectorFactory
import json

# Set seed for consistent language detection
DetectorFactory.seed = 0

class NLPPipeline:
    def __init__(self):
        # Load models
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        # Store details about last classification for audit/logging
        self.last_classification_info = {}
        
        # Load protest slang dictionary
        try:
            with open("data/protest_slang_dict.json", "r", encoding='utf-8') as f:
                self.protest_slang = json.load(f)
        except FileNotFoundError:
            # Fallback if file doesn't exist
            self.protest_slang = {
                "en": ["protest", "riot", "strike", "tear gas"],
                "hi": ["andolan", "hartal", "vidroh"],
                "bn": ["birodh", "andolon"],
                "ur": ["احتجاج", "ہڑتال"]
            }
        except UnicodeDecodeError:
            # Fallback if encoding issues
            print("Warning: Could not read protest_slang_dict.json due to encoding issues, using fallback")
            self.protest_slang = {
                "en": ["protest", "riot", "strike", "tear gas"],
                "hi": ["andolan", "hartal", "vidroh"],
                "bn": ["birodh", "andolon"],
                "ur": ["احتجاج", "ہڑتال"]
            }
        
        # Load language map
        try:
            with open("data/language_map.json", "r", encoding='utf-8') as f:
                self.language_map = json.load(f)
        except FileNotFoundError:
            # Fallback if file doesn't exist
            self.language_map = {
                "en": "English",
                "hi": "Hindi",
                "bn": "Bengali",
                "ur": "Urdu"
            }
        except UnicodeDecodeError:
            # Fallback if encoding issues
            print("Warning: Could not read language_map.json due to encoding issues, using fallback")
            self.language_map = {
                "en": "English",
                "hi": "Hindi",
                "bn": "Bengali",
                "ur": "Urdu"
            }
        
        # Initialize relevance classifier (using a simple keyword-based approach)
        self.protest_keywords = []
        for lang_keywords in self.protest_slang.values():
            self.protest_keywords.extend(lang_keywords)
        
        # Use Nominatim (OpenStreetMap) for geocoding - no API key needed
        self.geocoding_service = "nominatim"
        
        # Load spaCy English model for NER (if available)
        try:
            self.spacy_nlp = spacy.load('en_core_web_sm')
        except Exception as e:
            print(f"spaCy model not available: {e}")
            self.spacy_nlp = None

        # AI-based zero-shot classifier for protest relevance
        try:
            self.unrest_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except Exception as e:
            print(f"Zero-shot classifier not available: {e}")
            self.unrest_classifier = None

    def process(self, post: Dict) -> Dict:
        """Process a post through the full NLP pipeline"""
        content = post.get("content", "")
        
        # 1. Language detection (no translation for now - simplified)
        language = self.detect_language(content)
        
        # 2. Protest relevance classification
        protest_score = self.classify_protest_relevance(content)
        
        # 3. Named Entity Recognition
        entities = self.extract_entities(content)
        
        # 4. Sentiment analysis
        sentiment_score = self.analyze_sentiment(content)
        
        # 5. Geolocation extraction
        location_lat, location_lng = self.extract_geolocation(content, post.get("location_raw", ""))
        
        return {
            "raw_post_id": post.get("id"),
            "protest_score": protest_score,
            "sentiment_score": sentiment_score,
            "location_lat": location_lat,
            "location_lng": location_lng,
            "language": language,
            "platform": post.get("platform"),
            "link": post.get("link"),
            "entities": entities,
            "status": "unverified",
            "title": post.get("title"),
            "headline": post.get("headline"),
            "content": post.get("content"),
            # Provide classification details for downstream logging without changing DB schema
            "classification_info": self.last_classification_info
        }

    def detect_language(self, text: str) -> str:
        """Detect language of the text"""
        try:
            if not text or len(text.strip()) == 0:
                return "en"
            # Clean text and handle encoding issues
            clean_text = self.clean_text(text[:100])
            return detect(clean_text)
        except Exception as e:
            print(f"Language detection error: {e}")
            return "en"  # Default to English

    def clean_text(self, text: str) -> str:
        """Clean text and handle encoding issues"""
        try:
            # Handle different encodings
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            
            # Handle None or non-string types
            if not isinstance(text, str):
                text = str(text)
            
            # Remove or replace problematic characters more aggressively
            import re
            # Remove control characters and non-printable characters
            text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
            # Remove characters that can't be encoded properly
            text = re.sub(r'[^\x20-\x7E\u00A0-\uFFFF]', ' ', text)
            # Replace multiple spaces and newlines with single space
            text = re.sub(r'\s+', ' ', text)
            # Remove any remaining problematic characters
            text = ''.join(char for char in text if ord(char) < 65536)
            
            return text.strip()
        except Exception as e:
            print(f"Text cleaning error: {e}")
            return "text"

    def classify_protest_relevance(self, text: str) -> float:
        """Classify if text is protest-related (0.0 to 1.0) using enhanced filtering logic."""
        try:
            clean_text = self.clean_text(text).lower()
            # Reset last classification info
            self.last_classification_info = {"decision": "unknown", "details": {}}
            
            # Exclude non-relevant content early
            if self._is_irrelevant_content(clean_text):
                # Best-effort reason summary
                self.last_classification_info = {
                    "decision": "irrelevant_keywords",
                    "details": {
                        "note": "Matched irrelevant topic patterns (sports/entertainment/business/tech/weather)",
                    }
                }
                return 0.0
                
            if self.unrest_classifier:
                result = self.unrest_classifier(
                    clean_text,
                    candidate_labels=["protest", "riot", "civil unrest", "political demonstration", "strike", "normal news", "sports", "entertainment", "weather", "business"],
                    multi_label=True
                )
                # Return the score for the most relevant unrest label
                protest_labels = ["protest", "riot", "civil unrest", "political demonstration", "strike"]
                irrelevant_labels = ["sports", "entertainment", "weather", "business"]
                
                protest_scores = [score for label, score in zip(result['labels'], result['scores']) if label in protest_labels]
                irrelevant_scores = [score for label, score in zip(result['labels'], result['scores']) if label in irrelevant_labels]
                
                max_protest_score = max(protest_scores) if protest_scores else 0.0
                max_irrelevant_score = max(irrelevant_scores) if irrelevant_scores else 0.0
                
                # If irrelevant content is more confident, return 0
                if max_irrelevant_score > max_protest_score and max_irrelevant_score > 0.6:
                    self.last_classification_info = {
                        "decision": "zero_shot_irrelevant",
                        "details": {
                            "max_protest_score": max_protest_score,
                            "max_irrelevant_score": max_irrelevant_score
                        }
                    }
                    return 0.0
                    
                self.last_classification_info = {
                    "decision": "zero_shot_protest",
                    "details": {
                        "max_protest_score": max_protest_score,
                        "max_irrelevant_score": max_irrelevant_score
                    }
                }
                return max_protest_score
            else:
                # Enhanced fallback logic
                score = self._enhanced_keyword_scoring(clean_text)
                self.last_classification_info = {
                    "decision": "keyword_scoring",
                    "details": {"score": score}
                }
                return score
                
        except Exception as e:
            print(f"AI protest classification error: {e}")
            self.last_classification_info = {"decision": "error", "details": {"error": str(e)}}
            return 0.0
    
    def _is_irrelevant_content(self, text: str) -> bool:
        """Check if content is clearly not related to protests/civil unrest."""
        # Sports exclusions
        sports_keywords = ["football", "soccer", "basketball", "baseball", "cricket", "tennis", "golf", "hockey", 
                          "championship", "tournament", "league", "playoff", "olympics", "fifa", "nfl", "nba", 
                          "premier league", "bundesliga", "la liga", "serie a", "champions league", "world cup",
                          "score", "goal", "touchdown", "match", "game", "team", "player", "coach", "stadium",
                          "manchester united", "chelsea", "arsenal", "liverpool"]
        
        # Entertainment exclusions  
        entertainment_keywords = ["movie", "film", "cinema", "actor", "actress", "director", "tv show", "television",
                                "series", "netflix", "disney", "hollywood", "bollywood", "music", "concert", "album",
                                "song", "singer", "band", "celebrity", "award", "oscar", "grammy", "emmy", "festival",
                                "premiere", "engagement", "wedding"]
        
        # Business/Finance exclusions
        business_keywords = ["stock market", "shares", "trading", "investment", "cryptocurrency", "bitcoin", "crypto",
                           "earnings", "revenue", "profit", "merger", "acquisition", "ipo", "nasdaq", "dow jones",
                           "quarterly results", "financial report", "economic growth", "gdp", "price surges",
                           "monthly high", "investor optimism", "tech shares"]
        
        # Technology exclusions
        tech_keywords = ["smartphone", "iphone", "android", "software", "app", "update", "tech company", 
                        "artificial intelligence", "ai", "machine learning", "blockchain", "metaverse",
                        "camera", "battery life", "new phone", "device", "gadget"]
        
        # Weather/Natural disasters (unless they lead to unrest)
        weather_keywords = ["weather", "rain", "storm", "hurricane", "tornado", "flood", "earthquake", "tsunami",
                          "snowfall", "heatwave", "drought", "wildfire", "forecast", "expected"]
        
        all_irrelevant = sports_keywords + entertainment_keywords + business_keywords + weather_keywords + tech_keywords
        
        # Strong exclusion patterns - if any of these appear, likely irrelevant
        strong_exclusion_patterns = [
            "stock market", "bitcoin price", "cryptocurrency", "iphone", "smartphone", "movie premiere",
            "celebrity couple", "weather forecast", "recipe", "cooking", "premier league", "championship",
            "tech shares", "quarterly results", "new device", "battery life", "camera", "gaming"
        ]
        
        # Check for strong exclusion patterns first
        for pattern in strong_exclusion_patterns:
            if pattern in text:
                return True
        
        # Check if text contains multiple irrelevant keywords
        irrelevant_count = sum(1 for keyword in all_irrelevant if keyword in text)
        words = len(text.split())
        
        # If more than 8% of identifiable content is irrelevant, exclude (lowered threshold)
        if words > 0 and irrelevant_count / words > 0.08:
            return True
            
        # Check for specific irrelevant contexts
        if any(phrase in text for phrase in [
            "sports news", "entertainment news", "movie review", "stock price", "weather forecast",
            "celebrity gossip", "gaming", "video game", "recipe", "cooking", "fashion", "beauty",
            "health tips", "diet", "workout", "fitness", "travel", "vacation", "tourism",
            "tech company", "new release", "product launch"
        ]):
            return True
            
        return False
    
    def _enhanced_keyword_scoring(self, text: str) -> float:
        """Enhanced keyword-based scoring with weighted categories."""
        words = text.split()
        if len(words) == 0:
            return 0.0
            
        # Load enhanced keywords with weights
        protest_keywords = {
            # High-weight direct protest terms
            "protest": 3.0, "demonstration": 3.0, "rally": 2.5, "march": 2.0, "strike": 3.0,
            "riot": 4.0, "unrest": 4.0, "uprising": 3.5, "revolt": 3.5, "rebellion": 3.5,
            "protests": 3.0, "demonstrations": 3.0, "rallies": 2.5, "strikes": 3.0,
            
            # Medium-weight escalation terms
            "violence": 2.0, "clash": 2.5, "confrontation": 2.0, "police": 1.5, "arrest": 2.0,
            "tear gas": 3.0, "pepper spray": 2.5, "barricade": 2.5, "blockade": 2.5,
            "violent": 2.0, "clashes": 2.5, "arrests": 2.0,
            
            # Lower-weight early warning terms
            "gathering": 1.0, "crowd": 1.0, "tension": 1.5, "planned": 1.0, "organizing": 1.5,
            "mobilizing": 2.0, "activist": 1.5, "campaign": 1.0, "activists": 1.5,
            
            # Context-specific terms
            "civil rights": 2.0, "justice": 1.5, "freedom": 1.0, "democracy": 1.0, "oppression": 2.0,
            "government": 1.0, "authority": 1.0, "regime": 1.5, "dictatorship": 2.0,
            "policy": 1.0, "turns violent": 3.0, "against government": 2.0
        }
        
        # Calculate weighted score
        total_weight = 0.0
        
        # Check for multi-word phrases first
        multi_word_phrases = {
            "turns violent": 3.0, "against government": 2.0, "civil rights": 2.0,
            "tear gas": 3.0, "pepper spray": 2.5, "government policy": 1.5
        }
        
        for phrase, weight in multi_word_phrases.items():
            if phrase in text:
                total_weight += weight
        
        # Then check individual words
        for word in words:
            if word in protest_keywords:
                total_weight += protest_keywords[word]
        
        # Normalize by text length but give bonus for multiple relevant terms
        base_score = total_weight / len(words)
        
        # Bonus for multiple different protest terms
        unique_protest_words = len(set(words) & set(protest_keywords.keys()))
        multi_word_matches = sum(1 for phrase in multi_word_phrases.keys() if phrase in text)
        
        if unique_protest_words >= 2 or multi_word_matches >= 1:
            base_score *= 1.5
        if unique_protest_words >= 3 or multi_word_matches >= 2:
            base_score *= 2.0
            
        return min(base_score * 5.0, 1.0)  # Scale and cap at 1.0

    def extract_entities(self, text: str) -> Dict:
        """Extract named entities (locations, organizations, persons)"""
        entities = {
            "locations": [],
            "organizations": [],
            "persons": []
        }
        
        try:
            # Clean text first
            clean_text = self.clean_text(text)
            
            # Extract locations using geotext
            try:
                geo = geotext.GeoText(clean_text)
                entities["locations"] = list(set(geo.cities + geo.countries))
            except Exception as e:
                print(f"Geotext error: {e}")
            
            # Simple organization extraction (police, government, etc.)
            org_keywords = ["police", "government", "army", "military", "party", "ministry"]
            for keyword in org_keywords:
                if keyword.lower() in clean_text.lower():
                    entities["organizations"].append(keyword.title())
        except Exception as e:
            print(f"Entity extraction error: {e}")
        
        return entities

    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment using VADER"""
        try:
            clean_text = self.clean_text(text)
            scores = self.sentiment_analyzer.polarity_scores(clean_text)
            return scores["compound"]  # Returns -1 to 1
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 0.0

    def extract_geolocation(self, text: str, location_raw: str) -> tuple:
        """Extract latitude and longitude from text or raw location, using multiple methods"""
        try:
            clean_text = self.clean_text(text)
            locations = set()
            # 1. geotext cities/countries
            try:
                geo = geotext.GeoText(clean_text)
                locations.update(geo.cities)
                locations.update(geo.countries)
            except Exception as e:
                print(f"Geotext location extraction error: {e}")
            # 2. spaCy NER (if available)
            if self.spacy_nlp:
                try:
                    doc = self.spacy_nlp(clean_text)
                    for ent in doc.ents:
                        if ent.label_ in ["GPE", "LOC"]:
                            locations.add(ent.text)
                except Exception as e:
                    print(f"spaCy NER error: {e}")
            # 3. location_raw fallback
            if location_raw:
                clean_location = self.clean_text(location_raw)
                locations.add(clean_location)
            # Try geocoding all found locations, return first valid
            for loc in locations:
                latlng = self.geocode_location(loc)
                if latlng and latlng[0] is not None and latlng[1] is not None:
                    return latlng
        except Exception as e:
            print(f"Geolocation extraction error: {e}")
        return None, None

    def geocode_location(self, location: str) -> tuple:
        """Convert location string to lat/lng using Nominatim (OpenStreetMap)"""
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": location,
                "format": "json",
                "limit": 1,
                "addressdetails": 1
            }
            
            # Add User-Agent header (required by Nominatim)
            headers = {
                "User-Agent": "NOESIS_Bot/1.0 (https://github.com/your-repo)"
            }
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            if data and len(data) > 0:
                lat = float(data[0]["lat"])
                lng = float(data[0]["lon"])
                return lat, lng
        
        except Exception as e:
            print(f"Geocoding error: {e}")
        
        return None, None 