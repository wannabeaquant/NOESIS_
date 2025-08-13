import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { MapContainer, TileLayer, Circle, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// High-risk areas with pulsing animations
const highRiskAreas = [
  { lat: 40.7128, lng: -74.0060, severity: 'high', radius: 50000, name: 'New York' },
  { lat: 34.0522, lng: -118.2437, severity: 'high', radius: 45000, name: 'Los Angeles' },
  { lat: 51.5074, lng: -0.1278, severity: 'medium', radius: 40000, name: 'London' },
  { lat: 48.8566, lng: 2.3522, severity: 'high', radius: 35000, name: 'Paris' },
  { lat: 35.6762, lng: 139.6503, severity: 'medium', radius: 30000, name: 'Tokyo' },
  { lat: -33.8688, lng: 151.2093, severity: 'low', radius: 25000, name: 'Sydney' },
  { lat: 55.7558, lng: 37.6176, severity: 'high', radius: 55000, name: 'Moscow' },
  { lat: 39.9042, lng: 116.4074, severity: 'medium', radius: 50000, name: 'Beijing' },
  { lat: 19.0760, lng: 72.8777, severity: 'high', radius: 60000, name: 'Mumbai' },
  { lat: -23.5505, lng: -46.6333, severity: 'medium', radius: 40000, name: 'São Paulo' },
];

const AnimatedCircles = () => {
  const map = useMap();
  
  useEffect(() => {
    const circles = [];
    
    highRiskAreas.forEach((area, index) => {
      const color = area.severity === 'high' ? '#ef4444' : 
                   area.severity === 'medium' ? '#f59e0b' : '#10b981';
      
      // Create multiple pulsing circles for each area
      for (let i = 0; i < 3; i++) {
        const circle = L.circle([area.lat, area.lng], {
          radius: area.radius + (i * 10000),
          color: color,
          fillColor: color,
          fillOpacity: 0.1 - (i * 0.03),
          weight: 2,
          opacity: 0.6 - (i * 0.2),
        }).addTo(map);
        
        circles.push(circle);
        
        // Animate the circle
        let scale = 1;
        let growing = true;
        
        const animate = () => {
          if (growing) {
            scale += 0.01;
            if (scale >= 1.5) growing = false;
          } else {
            scale -= 0.01;
            if (scale <= 0.8) growing = true;
          }
          
          const newRadius = (area.radius + (i * 10000)) * scale;
          circle.setRadius(newRadius);
          
          setTimeout(animate, 50 + (i * 20));
        };
        
        setTimeout(() => animate(), index * 200);
      }
    });
    
    return () => {
      circles.forEach(circle => map.removeLayer(circle));
    };
  }, [map]);
  
  return null;
};

const AnimatedMapBackground = () => {
  return (
    <div className="absolute inset-0 z-0">
      <MapContainer
        center={[20, 0]}
        zoom={2}
        style={{ height: '100%', width: '100%' }}
        zoomControl={false}
        attributionControl={false}
        dragging={false}
        touchZoom={false}
        doubleClickZoom={false}
        scrollWheelZoom={false}
        boxZoom={false}
        keyboard={false}
        className="opacity-30"
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution=""
        />
        <AnimatedCircles />
      </MapContainer>
      
      {/* Overlay gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-gray-950/50 to-gray-950/80 pointer-events-none"></div>
      
      {/* Floating particles effect */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-blue-400 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              y: [0, -20, 0],
              opacity: [0.3, 1, 0.3],
              scale: [1, 1.5, 1],
            }}
            transition={{
              duration: 3 + Math.random() * 2,
              repeat: Infinity,
              ease: "easeInOut",
              delay: Math.random() * 2,
            }}
          />
        ))}
      </div>
    </div>
  );
};

export default AnimatedMapBackground; 