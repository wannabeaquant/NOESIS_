import requests
from typing import List, Dict
import os
from datetime import datetime, timedelta
import json
import random

class FinancialCollector:
    def __init__(self, api_key: str | None = None):
        if api_key is not None:
            self.api_key = api_key
        else:
            env_api_key = os.getenv("FINANCIAL_API_KEY")
            if env_api_key is None:
                raise ValueError("API key must be provided either as an argument or via the FINANCIAL_API_KEY environment variable.")
            self.api_key = env_api_key
        self.base_url = "https://api.financial-data.com/v1"  # Placeholder

    def collect(self) -> List[Dict]:
        """Collect financial data for economic unrest indicators"""
        posts = []
        
        # Simulate financial data collection
        mock_financial_data = self._get_mock_financial_data()
        
        for data in mock_financial_data:
            post = {
                "platform": "financial",
                "content": data["description"],
                "author": "Financial Monitor",
                "timestamp": data["timestamp"],
                "location_raw": data["region"],
                "link": f"https://financial.example.com/alert/{data['id']}",
                "extra": {
                    "market_volatility": data["market_volatility"],
                    "currency_fluctuation": data["currency_fluctuation"],
                    "commodity_prices": data["commodity_prices"],
                    "economic_confidence": data["economic_confidence"]
                }
            }
            posts.append(post)
            
        return posts
    
    def _get_mock_financial_data(self) -> List[Dict]:
        """Generate mock financial data for demonstration"""
        regions = ["Global", "Asia-Pacific", "Europe", "Americas", "Middle East"]
        commodities = ["Oil", "Gold", "Food", "Energy"]
        
        mock_data = []
        for i in range(4):
            region = random.choice(regions)
            market_volatility = random.uniform(0.1, 0.9)
            currency_fluctuation = random.uniform(-0.15, 0.15)
            commodity_price_change = random.uniform(-0.2, 0.2)
            economic_confidence = random.uniform(0.2, 0.9)
            
            # Determine if this indicates potential economic unrest
            if (market_volatility > 0.7 or abs(currency_fluctuation) > 0.1 or 
                abs(commodity_price_change) > 0.15 or economic_confidence < 0.4):
                description = f"Economic instability detected in {region}. High volatility and market stress indicators."
                severity = "high"
            else:
                description = f"Stable economic conditions in {region}. Normal market activity observed."
                severity = "low"
            
            mock_data.append({
                "id": f"fin_{i+1}",
                "description": description,
                "region": region,
                "timestamp": (datetime.utcnow() - timedelta(hours=random.randint(1, 12))).isoformat(),
                "market_volatility": market_volatility,
                "currency_fluctuation": currency_fluctuation,
                "commodity_prices": {
                    "oil": 1 + random.uniform(-0.1, 0.1),
                    "gold": 1 + random.uniform(-0.05, 0.05),
                    "food": 1 + random.uniform(-0.15, 0.15)
                },
                "economic_confidence": economic_confidence,
                "severity": severity
            })
        
        return mock_data 