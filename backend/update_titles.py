#!/usr/bin/env python3
"""
Script to update incident titles to use proper place names
instead of coordinate strings
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.database import SessionLocal
from app.models.incident import Incident

def update_incident_titles():
    """Update incident titles to use proper place names"""
    db = SessionLocal()
    
    try:
        # Get all incidents with coordinate-based titles
        incidents = db.query(Incident).filter(
            Incident.title.like("Civil unrest reported in Location (%")
        ).all()
        
        print(f"Found {len(incidents)} incidents with coordinate-based titles to update")
        
        updated_count = 0
        for i, incident in enumerate(incidents):
            print(f"Processing incident {i+1}/{len(incidents)}: {incident.incident_id}")
            # Use getattr/setattr to avoid assigning to the Column object
            location_val = getattr(incident, 'location', None)
            if location_val is not None and not (isinstance(location_val, str) and location_val.startswith("Location (")):
                # Update title to use the proper location name
                new_title = f"Civil unrest reported in {location_val}"
                setattr(incident, 'title', new_title)
                print(f"  Updated title: {new_title}")
                updated_count += 1
            else:
                print(f"  Location still needs geocoding: {location_val}")
        
        # Commit all changes
        db.commit()
        print(f"\n✅ Successfully updated {updated_count} incident titles!")
        
    except Exception as e:
        print(f"❌ Error updating incident titles: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("🚀 NOESIS Title Update Script")
    print("=" * 50)
    
    update_incident_titles() 