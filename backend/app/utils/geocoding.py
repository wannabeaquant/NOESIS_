import requests
import time
from typing import Optional, Tuple

class GeocodingService:
    def __init__(self):
        self.base_url = "https://nominatim.openstreetmap.org/reverse"
        self.headers = {
            'User-Agent': 'NOESIS/1.0 (https://github.com/your-repo)'
        }
        # Rate limiting: Nominatim allows 1 request per second
        self.last_request_time = 0
        self.min_interval = 1.0

    def get_place_name(self, lat: float, lng: float) -> Optional[str]:
        """
        Convert coordinates to a human-readable place name using Nominatim
        Returns None if geocoding fails
        """
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_interval:
                time.sleep(self.min_interval - time_since_last)
            
            params = {
                'lat': lat,
                'lon': lng,
                'format': 'json',
                'addressdetails': 1,
                'accept-language': 'en'
            }
            
            response = requests.get(
                self.base_url, 
                params=params, 
                headers=self.headers,
                timeout=10
            )
            
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                return self._extract_place_name(data)
            else:
                print(f"Geocoding failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Geocoding error: {e}")
            return None

    def _extract_place_name(self, data: dict) -> str:
        """
        Extract the most relevant place name from Nominatim response
        """
        address = data.get('address', {})
        
        # Try to get city, state, country hierarchy
        city = address.get('city') or address.get('town') or address.get('village')
        state = address.get('state') or address.get('province')
        country = address.get('country')
        
        if city and country:
            return f"{city}, {country}"
        elif city:
            return city
        elif state and country:
            return f"{state}, {country}"
        elif country:
            return country
        else:
            # Fallback to display_name if available
            display_name = data.get('display_name', '')
            if display_name:
                # Take first part of display_name (usually the most specific location)
                parts = display_name.split(',')
                return parts[0].strip()
            else:
                return "Unknown Location"

    def get_coordinates_from_place(self, place_name: str) -> Optional[Tuple[float, float]]:
        """
        Convert place name to coordinates (forward geocoding)
        Returns (lat, lng) tuple or None if geocoding fails
        """
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_interval:
                time.sleep(self.min_interval - time_since_last)
            
            params = {
                'q': place_name,
                'format': 'json',
                'limit': 1
            }
            
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params=params,
                headers=self.headers,
                timeout=10
            )
            
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    lat = float(data[0]['lat'])
                    lon = float(data[0]['lon'])
                    return (lat, lon)
            
            return None
            
        except Exception as e:
            print(f"Forward geocoding error: {e}")
            return None 