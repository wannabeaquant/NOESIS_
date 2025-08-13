import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { motion } from 'framer-motion';
import apiService from '../services/api';

// Fix for default markers in react-leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Mock incident data for preview
const mockIncidents = [
  {
    id: 1,
    title: "Climate Protest - London",
    location: "London, UK",
    lat: 51.5074,
    lng: -0.1278,
    severity: "medium",
    participants: 5000,
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    description: "Large climate change protest in central London",
    sentiment: "peaceful",
    verification: "verified",
    sources: ["Twitter", "BBC News"],
    tags: ["climate", "protest", "environment"]
  },
  {
    id: 2,
    title: "Labor Rights Demonstration",
    location: "New York, USA",
    lat: 40.7128,
    lng: -74.0060,
    severity: "high",
    participants: 12000,
    timestamp: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
    description: "Massive labor rights demonstration in Manhattan",
    sentiment: "tense",
    verification: "verified",
    sources: ["CNN", "Reuters"],
    tags: ["labor", "rights", "workers"]
  },
  {
    id: 3,
    title: "Student Protest - Paris",
    location: "Paris, France",
    lat: 48.8566,
    lng: 2.3522,
    severity: "medium",
    participants: 3000,
    timestamp: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
    description: "Student protest against education reforms",
    sentiment: "peaceful",
    verification: "pending",
    sources: ["Le Monde", "Social Media"],
    tags: ["education", "students", "reform"]
  },
  {
    id: 4,
    title: "Anti-Corruption Rally",
    location: "Mumbai, India",
    lat: 19.0760,
    lng: 72.8777,
    severity: "high",
    participants: 8000,
    timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    description: "Large anti-corruption rally in Mumbai",
    sentiment: "hostile",
    verification: "verified",
    sources: ["Times of India", "NDTV"],
    tags: ["corruption", "politics", "reform"]
  },
  {
    id: 5,
    title: "Housing Rights March",
    location: "São Paulo, Brazil",
    lat: -23.5505,
    lng: -46.6333,
    severity: "low",
    participants: 2000,
    timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
    description: "Peaceful march for housing rights",
    sentiment: "peaceful",
    verification: "verified",
    sources: ["Globo", "Local News"],
    tags: ["housing", "rights", "social"]
  }
];

// Custom marker icons based on severity
const createCustomIcon = (severity) => {
  const colors = {
    high: '#ef4444',
    medium: '#f59e0b',
    low: '#10b981'
  };
  
  const color = colors[severity] || '#6b7280';
  
  return L.divIcon({
    className: 'custom-marker',
    html: `
      <div style="
        background-color: ${color};
        width: 16px;
        height: 16px;
        border-radius: 50%;
        border: 3px solid white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        display: flex;
        align-items: center;
        justify-content: center;
        animation: pulse 2s infinite;
      ">
        <div style="
          width: 6px;
          height: 6px;
          background-color: white;
          border-radius: 50%;
        "></div>
      </div>
    `,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
    popupAnchor: [0, -11]
  });
};

const AnimatedMarkers = () => {
  const map = useMap();
  
  useEffect(() => {
    // Add custom CSS for pulse animation
    const style = document.createElement('style');
    style.textContent = `
      @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
      }
    `;
    document.head.appendChild(style);
    
    return () => {
      document.head.removeChild(style);
    };
  }, [map]);
  
  return null;
};

const InteractiveMap = () => {
  const [selectedIncident, setSelectedIncident] = useState(null);
  const [incidents, setIncidents] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchIncidents();
  }, []);

  const fetchIncidents = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await apiService.fetchLatestIncidents(20);
      setIncidents(data);
    } catch (err) {
      console.error('Error fetching incidents:', err);
      setError('Failed to load incident data');
    } finally {
      setIsLoading(false);
    }
  };

  const formatTimeAgo = (timestamp) => {
    if (!timestamp) return 'Unknown time';
    
    const now = new Date();
    const incidentTime = new Date(timestamp);
    const diffInHours = Math.floor((now - incidentTime) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return 'Less than 1 hour ago';
    if (diffInHours === 1) return '1 hour ago';
    return `${diffInHours} hours ago`;
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return 'text-red-400 bg-red-900/20';
      case 'medium': return 'text-yellow-400 bg-yellow-900/20';
      case 'low': return 'text-green-400 bg-green-900/20';
      default: return 'text-gray-400 bg-gray-900/20';
    }
  };

  // Filter out incidents without coordinates
  const incidentsWithCoords = incidents.filter(incident => 
    incident.location_lat && incident.location_lng
  );

  return (
    <div className="w-full h-full relative">
      <MapContainer
        center={[20, 0]}
        zoom={2}
        style={{ height: '100%', width: '100%' }}
        zoomControl={false}
        attributionControl={false}
        dragging={true}
        touchZoom={true}
        doubleClickZoom={false}
        scrollWheelZoom={false}
        boxZoom={false}
        keyboard={false}
        className="rounded-2xl"
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution=""
        />
        <AnimatedMarkers />
        
        {incidentsWithCoords.map((incident) => (
          <Marker
            key={incident.incident_id}
            position={[incident.location_lat, incident.location_lng]}
            icon={createCustomIcon(incident.severity)}
            eventHandlers={{
              click: () => setSelectedIncident(incident),
            }}
          >
            <Popup className="custom-popup" maxWidth={300}>
              <div className="bg-slate-900 text-white p-4 rounded-xl border border-slate-700 min-w-[280px]">
                {/* Header */}
                <div className="flex items-start justify-between mb-3">
                  <h3 className="text-lg font-bold text-white leading-tight pr-2">
                    {incident.title}
                  </h3>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium flex-shrink-0 ${getSeverityColor(incident.severity)}`}>
                    {incident.severity}
                  </span>
                </div>

                {/* Location and Time */}
                <div className="flex items-center text-slate-300 text-sm mb-3 space-x-4">
                  <div className="flex items-center">
                    <span className="mr-1">📍</span>
                    {incident.location || 'Unknown Location'}
                  </div>
                  <div className="flex items-center">
                    <span className="mr-1">⏰</span>
                    {formatTimeAgo(incident.timestamp)}
                  </div>
                </div>

                {/* Description */}
                <p className="text-slate-300 text-sm mb-3 leading-relaxed">
                  {incident.description || 'No description available'}
                </p>

                {/* Stats */}
                <div className="grid grid-cols-2 gap-4 mb-3">
                  <div className="flex items-center text-sm">
                    <span className="mr-2">📊</span>
                    <span className="text-slate-300">
                      {incident.sources?.length || 0} sources
                    </span>
                  </div>
                  <div className="flex items-center text-sm">
                    <span className="mr-2">🔍</span>
                    <span className="text-slate-300">
                      {incident.status}
                    </span>
                  </div>
                </div>

                {/* Verification Status */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center text-sm">
                    {incident.status === 'verified' ? (
                      <span className="mr-2">✅</span>
                    ) : (
                      <span className="mr-2">⏳</span>
                    )}
                    <span className="text-slate-300">
                      {incident.status}
                    </span>
                  </div>
                  <div className="text-xs text-slate-400">
                    ID: {incident.incident_id}
                  </div>
                </div>

                {/* Sources */}
                {incident.sources && incident.sources.length > 0 && (
                  <div className="border-t border-slate-700 pt-3">
                    <h4 className="text-sm font-semibold text-white mb-2">Sources:</h4>
                    <div className="space-y-2">
                      {incident.sources.slice(0, 3).map((source, index) => (
                        <a
                          key={index}
                          href={source}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="block px-2 py-1 bg-blue-900 text-blue-300 text-xs rounded hover:bg-blue-800/50 transition-colors break-all"
                        >
                          {source.includes('http') ? (
                            <span className="flex items-center">
                              <span className="truncate">{new URL(source).hostname}</span>
                              <span className="ml-1 text-blue-400">↗</span>
                            </span>
                          ) : (
                            source
                          )}
                        </a>
                      ))}
                      {incident.sources.length > 3 && (
                        <span className="px-2 py-1 bg-slate-700 text-slate-300 text-xs rounded-full">
                          +{incident.sources.length - 3} more
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
      
      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center rounded-2xl">
          <div className="text-center">
            <motion.div
              animate={{ 
                rotate: 360,
              }}
              transition={{ 
                duration: 20,
                repeat: Infinity,
                ease: "linear"
              }}
            >
              <span className="text-4xl">🌍</span>
            </motion.div>
            <p className="text-slate-300 text-lg mt-4">Loading Real-Time Data...</p>
          </div>
        </div>
      )}

      {/* Error overlay */}
      {error && (
        <div className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center rounded-2xl">
          <div className="text-center">
            <span className="text-4xl">⚠️</span>
            <p className="text-red-300 text-lg mt-4">{error}</p>
            <button
              onClick={fetchIncidents}
              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      )}
      {/* No data overlay */}
      {!isLoading && !error && incidentsWithCoords.length === 0 && (
        <div className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center rounded-2xl">
          <div className="text-center">
            <span className="text-4xl">📊</span>
            <p className="text-slate-300 mt-4">No incidents with location data</p>
            <p className="text-slate-400 text-sm mt-2">Try refreshing or check back later</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default InteractiveMap; 