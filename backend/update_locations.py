#!/usr/bin/env python3
"""
Script to update all incidents in the database with proper place names
using reverse geocoding instead of coordinate strings
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.database import SessionLocal
from app.models.incident import Incident
from app.utils.geocoding import GeocodingService
import time

def update_incident_locations():
    """Update all incidents with proper place names"""
    db = SessionLocal()
    geocoding_service = GeocodingService()
    
    try:
        # Get all incidents
        incidents = db.query(Incident).all()
        print(f"Found {len(incidents)} incidents to update")
        
        updated_count = 0
        for i, incident in enumerate(incidents):
            print(f"Processing incident {i+1}/{len(incidents)}: {incident.incident_id}")
            
            # Use getattr to safely access attributes
            location_val = getattr(incident, 'location', None)
            lat_val = getattr(incident, 'location_lat', None)
            lng_val = getattr(incident, 'location_lng', None)
            
            # Check if location is already a coordinate string
            if (location_val and 
                isinstance(location_val, str) and 
                location_val.startswith("Location (")):
                
                if lat_val is not None and lng_val is not None:
                    print(f"  Converting coordinates to place name...")
                    
                    # Get place name from coordinates
                    place_name = geocoding_service.get_place_name(
                        float(lat_val), 
                        float(lng_val)
                    )
                    
                    if place_name and place_name != "Unknown Location":
                        setattr(incident, 'location', place_name)
                        print(f"  Updated to: {place_name}")
                        updated_count += 1
                    else:
                        print(f"  Could not geocode coordinates")
                else:
                    print(f"  No coordinates available")
            else:
                print(f"  Already has place name: {location_val}")
            
            # Rate limiting for geocoding service
            time.sleep(1.1)  # Nominatim allows 1 request per second
        
        # Commit all changes
        db.commit()
        print(f"\n✅ Successfully updated {updated_count} incidents with proper place names!")
        
    except Exception as e:
        print(f"❌ Error updating incidents: {e}")
        db.rollback()
    finally:
        db.close()

def test_geocoding():
    """Test the geocoding service with some sample coordinates"""
    geocoding_service = GeocodingService()
    
    test_coords = [
        (40.7128, -74.0060),  # New York
        (51.5074, -0.1278),   # London
        (48.8566, 2.3522),    # Paris
        (35.6762, 139.6503),  # Tokyo
    ]
    
    print("Testing geocoding service...")
    for lat, lng in test_coords:
        place_name = geocoding_service.get_place_name(lat, lng)
        print(f"  ({lat}, {lng}) -> {place_name}")
        time.sleep(1.1)

if __name__ == "__main__":
    print("🚀 NOESIS Location Update Script")
    print("=" * 50)
    
    # Test geocoding first
    test_geocoding()
    print()
    
    # Update all incidents
    update_incident_locations() 