#!/usr/bin/env python3
"""
Real Machine Learning Models for NOESIS
Uses actual ML models for sentiment analysis and prediction
"""

import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

# Try to import ML libraries
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    import torch
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    logging.warning("ML libraries not available. Using fallback methods.")

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    probability: float
    confidence: float
    features: Dict[str, float]
    model_version: str
    timestamp: datetime

class MLPredictor:
    def __init__(self):
        self.has_ml_libs = HAS_ML_LIBS
        self.sentiment_pipeline = None
        self.protest_classifier = None
        self.anomaly_detector = None
        self.vectorizer = None
        self.scaler = None
        
        if self.has_ml_libs:
            self._load_models()
        else:
            logger.warning("Using fallback ML methods")
    
    def _load_models(self):
        """Load pre-trained ML models"""
        try:
            # Load sentiment analysis model
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Load protest detection classifier
            self._load_protest_classifier()
            
            # Load anomaly detector
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Load text vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Load feature scaler
            self.scaler = StandardScaler()
            
            logger.info("ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            self.has_ml_libs = False
    
    def _load_protest_classifier(self):
        """Load or train protest detection classifier"""
        try:
            # Try to load pre-trained model
            with open('models/protest_classifier.pkl', 'rb') as f:
                self.protest_classifier = pickle.load(f)
        except FileNotFoundError:
            # Train a new model with sample data
            self._train_protest_classifier()
    
    def _train_protest_classifier(self):
        """Train protest detection classifier with sample data"""
        # Sample training data (in real implementation, use actual protest data)
        protest_texts = [
            "protest against government policies",
            "demonstration for climate action",
            "rally for social justice",
            "march for human rights",
            "sit-in at university campus",
            "strike for better working conditions",
            "occupation of government building",
            "civil disobedience movement",
            "peaceful protest gathering",
            "mass demonstration downtown"
        ]
        
        non_protest_texts = [
            "weather forecast for today",
            "new restaurant opening downtown",
            "sports team wins championship",
            "movie review and ratings",
            "technology news update",
            "cooking recipe for dinner",
            "travel destination guide",
            "music concert announcement",
            "shopping mall opening hours",
            "traffic report for morning commute"
        ]
        
        # Create training data
        texts = protest_texts + non_protest_texts
        labels = [1] * len(protest_texts) + [0] * len(non_protest_texts)
        
        # Vectorize text
        X = self.vectorizer.fit_transform(texts)
        
        # Train classifier
        self.protest_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.protest_classifier.fit(X, labels)
        
        # Save model
        try:
            import os
            os.makedirs('models', exist_ok=True)
            with open('models/protest_classifier.pkl', 'wb') as f:
                pickle.dump(self.protest_classifier, f)
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using real ML model"""
        if not self.has_ml_libs or not self.sentiment_pipeline:
            return self._fallback_sentiment_analysis(text)
        
        try:
            # Use transformer model for sentiment analysis
            results = self.sentiment_pipeline(text)
            
            # Extract scores
            scores = results[0]
            sentiment_map = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive'
            }
            
            # Find highest scoring sentiment
            best_score = max(scores, key=lambda x: x['score'])
            sentiment = sentiment_map.get(best_score['label'], 'neutral')
            confidence = best_score['score']
            
            # Calculate sentiment score (-1 to 1)
            if sentiment == 'negative':
                sentiment_score = -confidence
            elif sentiment == 'positive':
                sentiment_score = confidence
            else:
                sentiment_score = 0.0
            
            return {
                'sentiment': sentiment,
                'score': sentiment_score,
                'confidence': confidence,
                'details': {s['label']: s['score'] for s in scores}
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._fallback_sentiment_analysis(text)
    
    def detect_protest_content(self, text: str) -> Dict[str, Any]:
        """Detect if text is protest-related using ML"""
        if not self.has_ml_libs or not self.protest_classifier:
            return self._fallback_protest_detection(text)
        
        try:
            # Vectorize text
            X = self.vectorizer.transform([text])
            
            # Predict
            prediction = self.protest_classifier.predict(X)[0]
            probability = self.protest_classifier.predict_proba(X)[0]
            
            return {
                'is_protest': bool(prediction),
                'probability': float(probability[1]),  # Probability of being protest
                'confidence': float(max(probability)),
                'features': self._extract_text_features(text)
            }
            
        except Exception as e:
            logger.error(f"Error in protest detection: {e}")
            return self._fallback_protest_detection(text)
    
    def predict_incident_likelihood(self, features: Dict[str, Any]) -> PredictionResult:
        """Predict likelihood of incident based on multiple features"""
        if not self.has_ml_libs:
            return self._fallback_prediction(features)
        
        try:
            # Extract numerical features
            feature_vector = self._extract_prediction_features(features)
            
            # Normalize features
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Use anomaly detection to identify unusual patterns
            anomaly_score = self.anomaly_detector.decision_function(feature_vector_scaled)[0]
            
            # Convert anomaly score to probability (higher anomaly = higher probability)
            probability = 1 / (1 + np.exp(-anomaly_score))
            
            # Calculate confidence based on feature quality
            confidence = self._calculate_confidence(features)
            
            return PredictionResult(
                probability=float(probability),
                confidence=float(confidence),
                features=features,
                model_version="v1.0",
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return self._fallback_prediction(features)
    
    def _extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract features from text"""
        text_lower = text.lower()
        
        # Protest-related keywords
        protest_keywords = [
            'protest', 'demonstration', 'rally', 'march', 'gathering',
            'unrest', 'disruption', 'blockade', 'occupation', 'strike',
            'civil disobedience', 'sit-in', 'walkout', 'boycott'
        ]
        
        # Violence-related keywords
        violence_keywords = [
            'violence', 'riot', 'clash', 'conflict', 'arrest',
            'police', 'tear gas', 'rubber bullets', 'barricade'
        ]
        
        # Peaceful keywords
        peaceful_keywords = [
            'peaceful', 'peace', 'unity', 'solidarity', 'justice',
            'non-violent', 'calm', 'orderly'
        ]
        
        features = {
            'protest_keyword_count': sum(1 for word in protest_keywords if word in text_lower),
            'violence_keyword_count': sum(1 for word in violence_keywords if word in text_lower),
            'peaceful_keyword_count': sum(1 for word in peaceful_keywords if word in text_lower),
            'text_length': len(text),
            'exclamation_count': text.count('!'),
            'hashtag_count': text.count('#'),
            'mention_count': text.count('@')
        }
        
        return features
    
    def _extract_prediction_features(self, features: Dict[str, Any]) -> List[float]:
        """Extract numerical features for prediction"""
        feature_vector = [
            features.get('social_media_volume', 0),
            features.get('news_coverage', 0),
            features.get('crowd_density', 0),
            features.get('police_activity', 0),
            features.get('traffic_anomaly', 0),
            features.get('weather_score', 0),
            features.get('sentiment_score', 0),
            features.get('protest_probability', 0),
            features.get('recent_incidents', 0),
            features.get('high_severity_count', 0)
        ]
        
        return feature_vector
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence in prediction based on data quality"""
        confidence_factors = []
        
        # Data source diversity
        sources = sum(1 for key in ['social_media_volume', 'news_coverage', 'crowd_density'] 
                     if features.get(key, 0) > 0)
        confidence_factors.append(min(sources / 3, 1.0))
        
        # Data recency (assume recent data is more reliable)
        confidence_factors.append(0.8)  # Placeholder
        
        # Data volume
        total_volume = features.get('social_media_volume', 0) + features.get('news_coverage', 0)
        confidence_factors.append(min(total_volume / 10, 1.0))
        
        return np.mean(confidence_factors)
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback sentiment analysis using simple rules"""
        text_lower = text.lower()
        
        negative_words = ['protest', 'demonstration', 'unrest', 'disruption', 'violence', 'arrest']
        positive_words = ['peaceful', 'peace', 'unity', 'solidarity', 'justice']
        
        negative_count = sum(1 for word in negative_words if word in text_lower)
        positive_count = sum(1 for word in positive_words if word in text_lower)
        
        if negative_count > positive_count:
            sentiment = 'negative'
            score = -0.5 - (negative_count * 0.1)
        elif positive_count > negative_count:
            sentiment = 'positive'
            score = 0.3 + (positive_count * 0.1)
        else:
            sentiment = 'neutral'
            score = 0.0
        
        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': 0.6,
            'details': {'fallback': True}
        }
    
    def _fallback_protest_detection(self, text: str) -> Dict[str, Any]:
        """Fallback protest detection using keyword matching"""
        text_lower = text.lower()
        protest_keywords = ['protest', 'demonstration', 'rally', 'march', 'gathering']
        
        keyword_count = sum(1 for word in protest_keywords if word in text_lower)
        is_protest = keyword_count > 0
        probability = min(keyword_count / 3, 1.0)
        
        return {
            'is_protest': is_protest,
            'probability': probability,
            'confidence': 0.7,
            'features': self._extract_text_features(text)
        }
    
    def _fallback_prediction(self, features: Dict[str, Any]) -> PredictionResult:
        """Fallback prediction using simple heuristics"""
        # Simple heuristic: more indicators = higher probability
        indicators = [
            features.get('social_media_volume', 0),
            features.get('news_coverage', 0),
            features.get('crowd_density', 0),
            features.get('recent_incidents', 0)
        ]
        
        probability = min(sum(indicators) / 10, 0.9)
        confidence = 0.6
        
        return PredictionResult(
            probability=probability,
            confidence=confidence,
            features=features,
            model_version="fallback",
            timestamp=datetime.utcnow()
        )

# Global ML predictor instance
ml_predictor = MLPredictor() 