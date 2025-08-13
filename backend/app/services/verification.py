from typing import List, Dict
from datetime import datetime, timedelta
from collections import defaultdict
import math
from app.utils.geocoding import GeocodingService

class VerificationService:
    def __init__(self):
        self.time_window = timedelta(minutes=60)  # 60-minute clustering window (loosened)
        self.distance_threshold = 200  # 200km radius for location clustering (loosened)
        self.geocoding_service = GeocodingService()

    def verify(self, processed_posts: List[Dict]) -> List[Dict]:
        """Verify and cluster processed posts into incidents with enhanced filtering"""
        # Strict filtering: only posts with significant protest scores
        relevant_posts = [post for post in processed_posts if post.get("protest_score", 0) > 0.35]
        
        print(f"[DEBUG] Filtered to {len(relevant_posts)} posts with protest_score > 0.35 (from {len(processed_posts)} total)")
        
        # Group posts by location and time
        clusters = self.cluster_posts(relevant_posts)
        
        # Create incidents from clusters with stricter requirements
        incidents = []
        for cluster in clusters:
            # Require at least 2 posts for an incident, unless it's a high-confidence single post
            if len(cluster) >= 2 or (len(cluster) == 1 and cluster[0].get("protest_score", 0) > 0.7):
                incident = self.create_incident(cluster)
                # Additional quality check - only create incidents with meaningful content
                if self._is_valid_incident(incident):
                    incidents.append(incident)
        
        print(f"[DEBUG] Created {len(incidents)} valid incidents from {len(clusters)} clusters")
        return incidents
    
    def _is_valid_incident(self, incident: Dict) -> bool:
        """Check if incident meets quality standards."""
        # Must have a meaningful title
        title = incident.get("title", "").strip()
        if len(title) < 10:
            return False
            
        # Must have valid location
        if not incident.get("location") or incident.get("location") == "Unknown Location":
            if not (incident.get("location_lat") and incident.get("location_lng")):
                return False
        
        # Must have sufficient confidence
        if incident.get("confidence_score", 0) < 30:
            return False
            
        # Check for spam patterns
        spam_indicators = ["click here", "visit our website", "buy now", "free gift", 
                          "limited time", "act now", "discount", "sale"]
        if any(spam in title.lower() for spam in spam_indicators):
            return False
            
        return True

    def cluster_posts(self, posts: List[Dict]) -> List[List[Dict]]:
        """Cluster posts by location and time proximity"""
        clusters = []
        processed = set()
        
        for i, post in enumerate(posts):
            if i in processed:
                continue
            
            cluster = [post]
            processed.add(i)
            
            # Find posts within time and location proximity
            for j, other_post in enumerate(posts[i+1:], i+1):
                if j in processed:
                    continue
                
                if self.are_posts_proximate(post, other_post):
                    cluster.append(other_post)
                    processed.add(j)
            
            if len(cluster) > 0:
                clusters.append(cluster)
        
        return clusters

    def are_posts_proximate(self, post1: Dict, post2: Dict) -> bool:
        """Check if two posts are proximate in time and location"""
        # Check time proximity
        time1 = datetime.fromisoformat(post1.get("timestamp", "").replace("Z", "+00:00"))
        time2 = datetime.fromisoformat(post2.get("timestamp", "").replace("Z", "+00:00"))
        
        if abs((time1 - time2).total_seconds()) > self.time_window.total_seconds():
            return False
        
        # Check location proximity
        lat1, lng1 = post1.get("location_lat"), post1.get("location_lng")
        lat2, lng2 = post2.get("location_lat"), post2.get("location_lng")
        
        if lat1 and lng1 and lat2 and lng2:
            distance = self.calculate_distance(lat1, lng1, lat2, lng2)
            return distance <= self.distance_threshold
        
        # If no coordinates, check if they're from the same platform (basic clustering)
        return post1.get("platform") == post2.get("platform")

    def calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c

    def create_incident(self, cluster: List[Dict]) -> Dict:
        """Create an incident from a cluster of posts with enhanced severity classification"""
        # Calculate average location
        lats = [post.get("location_lat") for post in cluster if post.get("location_lat")]
        lngs = [post.get("location_lng") for post in cluster if post.get("location_lng")]
        
        avg_lat = sum(lats) / len(lats) if lats else None
        avg_lng = sum(lngs) / len(lngs) if lngs else None
        
        # Enhanced severity determination
        severity, status = self._determine_enhanced_severity(cluster)
        
        # Find the most relevant post (highest protest score)
        most_relevant = max(cluster, key=lambda x: x.get("protest_score", 0))

        # Try to get a meaningful title
        incident_title = (
            most_relevant.get("title") or
            most_relevant.get("headline") or
            most_relevant.get("content") or
            f"Civil unrest reported in {self.get_location_name(avg_lat, avg_lng)}"
        )

        # For description, use the top 3 posts' content/title/headline
        sorted_cluster = sorted(cluster, key=lambda x: x.get("protest_score", 0), reverse=True)
        description_snippets = []
        for post in sorted_cluster[:3]:  # Top 3 posts
            snippet = post.get("content") or post.get("title") or post.get("headline")
            if snippet:
                description_snippets.append(snippet.strip())
        incident_description = " | ".join(description_snippets) if description_snippets else "No further details available."

        # For sources, use only valid, non-empty, http links from the top 3 posts by protest score, deduplicated
        sorted_cluster = sorted(cluster, key=lambda x: x.get("protest_score", 0), reverse=True)
        valid_links = []
        for post in sorted_cluster[:3]:
            link = post.get("link")
            if link and isinstance(link, str) and link.startswith("http") and link not in valid_links:
                valid_links.append(link)

        return {
            "title": incident_title.strip(),
            "description": incident_description,
            "sources": valid_links,
            "location": self.get_location_name(avg_lat, avg_lng),
            "location_lat": avg_lat,
            "location_lng": avg_lng,
            "severity": severity,
            "status": status,
            "confidence_score": self.calculate_confidence(cluster),
            "platform_diversity": len(set(post.get("platform") for post in cluster)),
            "source_count": len(cluster)
        }
    
    def _determine_enhanced_severity(self, cluster: List[Dict]) -> tuple:
        """Enhanced severity determination based on multiple factors."""
        # Content analysis
        severity_indicators = self._analyze_severity_indicators(cluster)
        
        # Source diversity and volume
        platform_diversity = len(set(post.get("platform") for post in cluster))
        source_count = len(cluster)
        
        # Average protest score and sentiment
        avg_protest_score = sum(post.get("protest_score", 0) for post in cluster) / len(cluster)
        avg_sentiment = sum(post.get("sentiment_score", 0) for post in cluster) / len(cluster)
        
        # Calculate severity score
        severity_score = 0
        
        # Content severity indicators (0-40 points)
        severity_score += severity_indicators["violence_level"] * 10
        severity_score += severity_indicators["scale_level"] * 8
        severity_score += severity_indicators["urgency_level"] * 6
        severity_score += severity_indicators["authority_response"] * 8
        severity_score += severity_indicators["casualty_level"] * 8
        
        # Source reliability (0-25 points)
        if source_count >= 5:
            severity_score += 15
        elif source_count >= 3:
            severity_score += 10
        elif source_count >= 2:
            severity_score += 5
            
        if platform_diversity >= 3:
            severity_score += 10
        elif platform_diversity >= 2:
            severity_score += 5
        
        # Protest relevance (0-20 points)
        severity_score += min(avg_protest_score * 20, 20)
        
        # Sentiment factor (0-15 points)
        if avg_sentiment < -0.5:  # Very negative
            severity_score += 15
        elif avg_sentiment < -0.2:  # Negative
            severity_score += 10
        elif avg_sentiment < 0:  # Slightly negative
            severity_score += 5
        
        # Determine final severity and status
        if severity_score >= 80:  # Raised threshold for high severity
            return "high", "verified" if platform_diversity >= 2 else "medium"
        elif severity_score >= 50:  # Raised threshold for medium severity
            return "medium", "verified" if platform_diversity >= 2 and source_count >= 2 else "medium"
        elif severity_score >= 30:  # Raised threshold for low severity
            return "low", "medium" if source_count >= 2 else "unverified"
        else:
            return "low", "unverified"
    
    def _analyze_severity_indicators(self, cluster: List[Dict]) -> Dict:
        """Analyze content for severity indicators."""
        # Combine all text content
        all_text = " ".join([
            (post.get("content") or "") + " " + 
            (post.get("title") or "") + " " + 
            (post.get("headline") or "")
            for post in cluster
        ]).lower()
        
        indicators = {
            "violence_level": 0,      # 0-4 scale
            "scale_level": 0,         # 0-4 scale  
            "urgency_level": 0,       # 0-4 scale
            "authority_response": 0,  # 0-4 scale
            "casualty_level": 0       # 0-4 scale
        }
        
        # Violence level indicators
        high_violence = ["shooting", "gunfire", "gunshot", "killed", "death", "murder", "bomb", "explosion", "fire", "burning", "looting", "vandalism"]
        medium_violence = ["violence", "violent", "clash", "fighting", "assault", "injured", "wounded", "hospital"]
        low_violence = ["scuffle", "push", "shove", "minor injuries", "peaceful protest turned"]
        
        if any(term in all_text for term in high_violence):
            indicators["violence_level"] = 4
        elif any(term in all_text for term in medium_violence):
            indicators["violence_level"] = 3 if sum(1 for term in medium_violence if term in all_text) >= 2 else 2
        elif any(term in all_text for term in low_violence):
            indicators["violence_level"] = 1
        
        # Scale level indicators
        large_scale = ["thousands", "hundreds", "massive", "huge crowd", "city-wide", "nationwide", "multiple cities", "across the country"]
        medium_scale = ["dozens", "many people", "large crowd", "significant", "widespread"]
        small_scale = ["few people", "small group", "handful", "isolated"]
        
        if any(term in all_text for term in large_scale):
            indicators["scale_level"] = 4
        elif any(term in all_text for term in medium_scale):
            indicators["scale_level"] = 3
        elif any(term in all_text for term in small_scale):
            indicators["scale_level"] = 1
        else:
            indicators["scale_level"] = 2  # Default medium
        
        # Urgency level indicators
        high_urgency = ["breaking", "urgent", "emergency", "immediate", "now", "currently", "ongoing", "live"]
        medium_urgency = ["developing", "latest", "update", "recent", "today"]
        
        if any(term in all_text for term in high_urgency):
            indicators["urgency_level"] = 4
        elif any(term in all_text for term in medium_urgency):
            indicators["urgency_level"] = 2
        else:
            indicators["urgency_level"] = 1
        
        # Authority response indicators
        heavy_response = ["military", "army", "troops", "martial law", "curfew", "state of emergency", "tear gas", "water cannon"]
        medium_response = ["police", "arrest", "detained", "custody", "riot police", "swat"]
        light_response = ["security", "monitoring", "watching", "increased presence"]
        
        if any(term in all_text for term in heavy_response):
            indicators["authority_response"] = 4
        elif any(term in all_text for term in medium_response):
            indicators["authority_response"] = 3
        elif any(term in all_text for term in light_response):
            indicators["authority_response"] = 2
        
        # Casualty level indicators
        high_casualties = ["multiple deaths", "many killed", "dozens injured", "mass casualties"]
        medium_casualties = ["death", "killed", "died", "fatality", "several injured"]
        low_casualties = ["injured", "wounded", "hurt", "hospitalized"]
        
        if any(term in all_text for term in high_casualties):
            indicators["casualty_level"] = 4
        elif any(term in all_text for term in medium_casualties):
            indicators["casualty_level"] = 3
        elif any(term in all_text for term in low_casualties):
            indicators["casualty_level"] = 2
        
        return indicators

    def get_location_name(self, lat: float, lng: float) -> str:
        """Get location name from coordinates using reverse geocoding"""
        if lat is None or lng is None:
            return "Unknown Location"
        
        # Try to get human-readable place name
        place_name = self.geocoding_service.get_place_name(lat, lng)
        if place_name and place_name != "Unknown Location":
            return place_name
        
        # Fallback to coordinates if geocoding fails
        return f"Location ({lat:.2f}, {lng:.2f})"

    def calculate_confidence(self, cluster: List[Dict]) -> int:
        """Calculate enhanced confidence score (0-100)"""
        platform_diversity = len(set(post.get("platform") for post in cluster))
        post_count = len(cluster)
        avg_protest_score = sum(post.get("protest_score", 0) for post in cluster) / len(cluster)
        
        # Base score from protest relevance (0-40 points)
        confidence = int(avg_protest_score * 40)
        
        # Source count factor (0-25 points)
        if post_count >= 5:
            confidence += 25
        elif post_count >= 3:
            confidence += 20
        elif post_count >= 2:
            confidence += 15
        else:
            confidence += 5
        
        # Platform diversity factor (0-20 points)
        if platform_diversity >= 3:
            confidence += 20
        elif platform_diversity >= 2:
            confidence += 15
        else:
            confidence += 5
        
        # Time clustering factor (0-10 points)
        if len(cluster) > 1:
            timestamps = []
            for post in cluster:
                try:
                    timestamp = datetime.fromisoformat(post.get("timestamp", "").replace("Z", "+00:00"))
                    timestamps.append(timestamp)
                except:
                    continue
            
            if len(timestamps) > 1:
                time_span = max(timestamps) - min(timestamps)
                # Closer timestamps indicate related events
                if time_span.total_seconds() < 3600:  # Within 1 hour
                    confidence += 10
                elif time_span.total_seconds() < 7200:  # Within 2 hours
                    confidence += 5
        
        # Content quality factor (0-5 points)
        # Check if posts have meaningful content
        meaningful_posts = sum(1 for post in cluster 
                             if len((post.get("content") or "").strip()) > 50)
        if meaningful_posts == len(cluster):
            confidence += 5
        elif meaningful_posts >= len(cluster) * 0.7:
            confidence += 3
        
        return min(confidence, 100) 