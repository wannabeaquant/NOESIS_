import requests
from typing import List, Dict
import os
from datetime import datetime, timedelta
import json
import random

class SatelliteCollector:
    def __init__(self, api_key: str = ""):  # Default to empty string
        self.api_key = api_key or os.getenv("SATELLITE_API_KEY") or ""
        self.base_url = "https://api.satellite-imagery.com/v1"  # Placeholder
        
    def collect(self) -> List[Dict]:
        """Collect satellite imagery data for crowd detection and infrastructure monitoring"""
        posts = []
        
        # Simulate satellite data collection
        mock_satellite_data = self._get_mock_satellite_data()
        
        for data in mock_satellite_data:
            post = {
                "platform": "satellite",
                "content": data["description"],
                "author": "Satellite System",
                "timestamp": data["timestamp"],
                "location_raw": data["location"],
                "link": f"https://satellite.example.com/image/{data['id']}",
                "extra": {
                    "crowd_density": data["crowd_density"],
                    "infrastructure_status": data["infrastructure_status"],
                    "image_coordinates": data["coordinates"],
                    "confidence_score": data["confidence"]
                }
            }
            posts.append(post)
            
        return posts
    
    def _get_mock_satellite_data(self) -> List[Dict]:
        """Generate mock satellite data for demonstration"""
        locations = [
            "Downtown Area", "City Center", "Government District", 
            "University Campus", "Shopping District", "Transport Hub"
        ]
        
        mock_data = []
        for i in range(5):
            location = random.choice(locations)
            crowd_density = random.uniform(0.1, 0.9)
            infrastructure_status = random.choice(["normal", "damaged", "congested"])
            
            # Determine if this indicates potential unrest
            if crowd_density > 0.7 and infrastructure_status in ["congested", "damaged"]:
                description = f"High crowd density detected in {location}. Infrastructure showing signs of stress."
                confidence = random.uniform(0.7, 0.95)
            else:
                description = f"Normal activity levels in {location}. Infrastructure status: {infrastructure_status}."
                confidence = random.uniform(0.5, 0.8)
            
            mock_data.append({
                "id": f"sat_{i+1}",
                "description": description,
                "location": location,
                "timestamp": (datetime.utcnow() - timedelta(hours=random.randint(1, 6))).isoformat(),
                "crowd_density": crowd_density,
                "infrastructure_status": infrastructure_status,
                "coordinates": {
                    "lat": random.uniform(28.0, 29.0),
                    "lng": random.uniform(76.0, 77.0)
                },
                "confidence": confidence
            })
        
        return mock_data 