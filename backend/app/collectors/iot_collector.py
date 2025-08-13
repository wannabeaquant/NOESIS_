import requests
from typing import List, Dict
import os
from datetime import datetime, timedelta
import json
import random

class IoTCollector:
    def __init__(self, api_key: str = ""):  # Default to empty string
        self.api_key = api_key or os.getenv("IOT_API_KEY") or ""
        self.base_url = "https://api.iot-sensors.com/v1"  # Placeholder
        
    def collect(self) -> List[Dict]:
        """Collect IoT sensor data for crowd density, traffic, and environmental monitoring"""
        posts = []
        
        # Simulate IoT data collection
        mock_iot_data = self._get_mock_iot_data()
        
        for data in mock_iot_data:
            post = {
                "platform": "iot",
                "content": data["description"],
                "author": "IoT Sensor Network",
                "timestamp": data["timestamp"],
                "location_raw": data["location"],
                "link": f"https://iot.example.com/sensor/{data['id']}",
                "extra": {
                    "crowd_density": data["crowd_density"],
                    "traffic_congestion": data["traffic_congestion"],
                    "air_quality": data["air_quality"],
                    "noise_levels": data["noise_levels"],
                    "sensor_confidence": data["sensor_confidence"]
                }
            }
            posts.append(post)
            
        return posts
    
    def _get_mock_iot_data(self) -> List[Dict]:
        """Generate mock IoT sensor data for demonstration"""
        locations = [
            "Central Square", "Government Building", "University Campus", 
            "Shopping Mall", "Transport Hub", "Residential Area"
        ]
        
        mock_data = []
        for i in range(6):
            location = random.choice(locations)
            crowd_density = random.uniform(0.1, 1.0)
            traffic_congestion = random.uniform(0.1, 1.0)
            air_quality = random.uniform(0.3, 1.0)  # 0.3 = poor, 1.0 = excellent
            noise_levels = random.uniform(0.1, 1.0)  # 0.1 = quiet, 1.0 = very loud
            sensor_confidence = random.uniform(0.7, 0.98)
            
            # Determine if this indicates potential unrest
            if (crowd_density > 0.8 or traffic_congestion > 0.8 or 
                air_quality < 0.5 or noise_levels > 0.8):
                description = f"Anomalous sensor readings in {location}. High activity and environmental stress detected."
                alert_level = "high"
            else:
                description = f"Normal sensor readings in {location}. Standard activity levels maintained."
                alert_level = "low"
            
            mock_data.append({
                "id": f"iot_{i+1}",
                "description": description,
                "location": location,
                "timestamp": (datetime.utcnow() - timedelta(minutes=random.randint(5, 60))).isoformat(),
                "crowd_density": crowd_density,
                "traffic_congestion": traffic_congestion,
                "air_quality": air_quality,
                "noise_levels": noise_levels,
                "sensor_confidence": sensor_confidence,
                "alert_level": alert_level
            })
        
        return mock_data 