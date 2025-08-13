#!/usr/bin/env python3
"""
Real Machine Learning Models for NOESIS
Uses actual ML models for sentiment analysis and prediction
"""

import json
import pickle
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

# Try to import ML libraries (simplified for Render)
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
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
            "Protest against government policy",
            "Demonstration in the city center",
            "Rally for civil rights",
            "Strike by workers",
            "Violent clashes with police",
            "Mass gathering for political change",
            "March against corruption",
            "Civil unrest in the capital",
            "Riot in downtown area",
            "Uprising against the regime"
        ]
        
        non_protest_texts = [
            "New movie released today",
            "Football match results",
            "Weather forecast for tomorrow",
            "Stock market update",
            "Celebrity wedding news",
            "Technology product launch",
            "Sports championship results",
            "Entertainment industry news",
            "Business quarterly report",
            "Travel destination guide"
        ]
        
        # Create training data
        X = protest_texts + non_protest_texts
        y = [1] * len(protest_texts) + [0] * len(non_protest_texts)
        
        # Vectorize text
        X_vectorized = self.vectorizer.fit_transform(X)
        
        # Train classifier
        self.protest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.protest_classifier.fit(X_vectorized, y)
        
        # Save the model
        try:
            import os
            os.makedirs('models', exist_ok=True)
            with open('models/protest_classifier.pkl', 'wb') as f:
                pickle.dump(self.protest_classifier, f)
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
    
    def predict_protest_probability(self, text: str) -> float:
        """Predict probability that text is protest-related"""
        if not self.has_ml_libs or not self.protest_classifier:
            # Fallback to simple keyword-based approach
            return self._fallback_protest_prediction(text)
        
        try:
            # Vectorize text
            X_vectorized = self.vectorizer.transform([text])
            
            # Get prediction probability
            prob = self.protest_classifier.predict_proba(X_vectorized)[0][1]
            return float(prob)
            
        except Exception as e:
            logger.error(f"Error in protest prediction: {e}")
            return self._fallback_protest_prediction(text)
    
    def _fallback_protest_prediction(self, text: str) -> float:
        """Simple keyword-based fallback for protest prediction"""
        protest_keywords = [
            "protest", "demonstration", "rally", "march", "strike", "riot", "unrest",
            "uprising", "revolt", "rebellion", "violence", "clash", "confrontation",
            "police", "arrest", "tear gas", "pepper spray", "barricade", "blockade",
            "gathering", "crowd", "tension", "organizing", "mobilizing", "activist",
            "civil rights", "justice", "freedom", "democracy", "oppression",
            "government", "authority", "regime", "dictatorship"
        ]
        
        text_lower = text.lower()
        matches = sum(1 for keyword in protest_keywords if keyword in text_lower)
        
        # Simple scoring: more matches = higher probability
        if matches == 0:
            return 0.0
        elif matches == 1:
            return 0.3
        elif matches == 2:
            return 0.6
        else:
            return min(0.9, 0.6 + (matches - 2) * 0.1)
    
    def detect_anomaly(self, features: List[float]) -> bool:
        """Detect if features represent an anomaly"""
        if not self.has_ml_libs or not self.anomaly_detector:
            return False
        
        try:
            # Reshape features for sklearn
            X = np.array(features).reshape(1, -1)
            
            # Fit and predict (for simplicity, fit on the fly)
            # In production, you'd want to pre-fit the model
            self.anomaly_detector.fit(X)
            prediction = self.anomaly_detector.predict(X)
            
            # -1 indicates anomaly, 1 indicates normal
            return prediction[0] == -1
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return False
    
    def extract_features(self, text: str, metadata: Dict) -> List[float]:
        """Extract numerical features from text and metadata"""
        features = []
        
        # Text-based features
        features.append(len(text))  # Text length
        features.append(len(text.split()))  # Word count
        features.append(len([c for c in text if c.isupper()]))  # Uppercase count
        
        # Protest keyword density
        protest_keywords = ["protest", "demonstration", "rally", "strike", "riot", "unrest"]
        protest_count = sum(1 for keyword in protest_keywords if keyword.lower() in text.lower())
        features.append(protest_count)
        
        # Metadata features
        features.append(metadata.get("followers_count", 0))
        features.append(metadata.get("retweet_count", 0))
        features.append(metadata.get("like_count", 0))
        
        # Time-based features (if available)
        if "created_at" in metadata:
            try:
                created_time = pd.to_datetime(metadata["created_at"])
                features.append(created_time.hour)  # Hour of day
                features.append(created_time.weekday())  # Day of week
            except:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
        
        return features
    
    def predict_incident_probability(self, text: str, metadata: Dict) -> PredictionResult:
        """Predict probability of incident based on text and metadata"""
        try:
            # Extract features
            features = self.extract_features(text, metadata)
            
            # Get protest probability
            protest_prob = self.predict_protest_probability(text)
            
            # Detect anomaly
            is_anomaly = self.detect_anomaly(features)
            
            # Combine predictions
            if is_anomaly:
                final_prob = min(1.0, protest_prob * 1.5)  # Boost probability for anomalies
            else:
                final_prob = protest_prob
            
            # Calculate confidence based on feature quality
            confidence = min(1.0, len([f for f in features if f > 0]) / len(features))
            
            return PredictionResult(
                probability=final_prob,
                confidence=confidence,
                features=dict(zip([f"feature_{i}" for i in range(len(features))], features)),
                model_version="1.0-simplified",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in incident prediction: {e}")
            return PredictionResult(
                probability=0.0,
                confidence=0.0,
                features={},
                model_version="1.0-fallback",
                timestamp=datetime.now()
            ) 