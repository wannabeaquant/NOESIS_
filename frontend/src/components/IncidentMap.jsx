import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Clock, Users, MapPin, Shield, AlertTriangle, CheckCircle, ExternalLink, X } from 'lucide-react';

// Fix for default markers in react-leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

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
        width: 20px;
        height: 20px;
        border-radius: 50%;
        border: 3px solid white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
        justify-content: center;
      ">
        <div style="
          width: 8px;
          height: 8px;
          background-color: white;
          border-radius: 50%;
        "></div>
      </div>
    `,
    iconSize: [26, 26],
    iconAnchor: [13, 13],
    popupAnchor: [0, -13]
  });
};

const IncidentMap = ({ selectedFilter = 'all', incidents = [] }) => {
  const [filteredIncidents, setFilteredIncidents] = useState(incidents);
  const [selectedSources, setSelectedSources] = useState(null);

  useEffect(() => {
    let filtered = incidents;
    
    switch (selectedFilter) {
      case 'high':
        filtered = incidents.filter(incident => incident.severity === 'high');
        break;
      case 'medium':
        filtered = incidents.filter(incident => incident.severity === 'medium');
        break;
      case 'low':
        filtered = incidents.filter(incident => incident.severity === 'low');
        break;
      case 'verified':
        filtered = incidents.filter(incident => incident.status === 'verified');
        break;
      case 'pending':
        filtered = incidents.filter(incident => incident.status === 'unverified');
        break;
      default:
        filtered = incidents;
    }
    
    setFilteredIncidents(filtered);
  }, [selectedFilter, incidents]);

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

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'hostile': return <AlertTriangle className="h-4 w-4 text-red-400" />;
      case 'tense': return <Shield className="h-4 w-4 text-yellow-400" />;
      case 'peaceful': return <CheckCircle className="h-4 w-4 text-green-400" />;
      default: return <AlertTriangle className="h-4 w-4 text-gray-400" />;
    }
  };

  // Filter out incidents without coordinates
  const incidentsWithCoords = filteredIncidents.filter(incident => 
    incident.location_lat && incident.location_lng
  );

  return (
    <MapContainer
      center={[40.7128, -74.0060]}
      zoom={2}
      style={{ height: '100%', width: '100%' }}
      className="rounded-lg"
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      
      {incidentsWithCoords.map((incident) => (
        <Marker
          key={incident.incident_id}
          position={[incident.location_lat, incident.location_lng]}
          icon={createCustomIcon(incident.severity)}
        >
          <Popup className="custom-popup" maxWidth={350}>
            <div className="bg-gray-900 text-white p-4 rounded-lg border border-gray-700 min-w-[220px] max-w-[320px] max-h-[300px] sm:max-h-[220px] xs:max-h-[160px] overflow-y-auto text-sm break-words">
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
              <div className="flex items-center text-gray-300 text-sm mb-3 space-x-4">
                <div className="flex items-center">
                  <MapPin className="h-4 w-4 mr-1 text-blue-400" />
                  {incident.location || 'Unknown Location'}
                </div>
              </div>

              {/* Description */}
              <p className="text-gray-300 text-sm mb-3 leading-relaxed">
                {incident.description || 'No description available'}
              </p>

              {/* Stats */}
              <div className="grid grid-cols-2 gap-4 mb-3">
                <div className="flex items-center text-sm">
                  <Users className="h-4 w-4 mr-2 text-purple-400" />
                  <span className="text-gray-300">
                    {incident.sources?.length || 0} sources
                  </span>
                </div>
                <div className="flex items-center text-sm">
                  <Shield className="h-4 w-4 text-blue-400" />
                  <span className="text-gray-300 ml-2 capitalize">
                    {incident.status}
                  </span>
                </div>
              </div>

              {/* Verification Status */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center text-sm">
                  {incident.status === 'verified' ? (
                    <CheckCircle className="h-4 w-4 mr-2 text-green-400" />
                  ) : (
                    <Clock className="h-4 w-4 mr-2 text-yellow-400" />
                  )}
                  <span className="text-gray-300 capitalize">
                    {incident.status}
                  </span>
                </div>
                <div className="text-xs text-gray-400">
                  ID: {incident.incident_id}
                </div>
              </div>

              {/* Sources */}
              {incident.sources && incident.sources.length > 0 && (
                <div className="border-t border-gray-700 pt-3">
                  <h4 className="text-sm font-semibold text-white mb-2">Sources:</h4>
                  <div className="space-y-2">
                    {incident.sources.slice(0, 3).map((source, index) => (
                      <a
                        key={index}
                        href={source}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block px-2 py-1 bg-blue-900/30 text-blue-300 text-xs rounded hover:bg-blue-800/50 transition-colors break-all"
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
                      <button
                        onClick={() => setSelectedSources(incident.sources)}
                        className="px-2 py-1 bg-gray-700 text-gray-300 text-xs rounded-full hover:bg-gray-600 transition-colors"
                      >
                        View all {incident.sources.length} sources
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          </Popup>
        </Marker>
      ))}

      {/* Sources Modal */}
      {selectedSources && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
          onClick={() => setSelectedSources(null)}
        >
          <div 
            className="bg-gray-900 rounded-lg max-w-2xl w-full max-h-[80vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-4 border-b border-gray-700">
              <h3 className="text-lg font-semibold text-white">All Sources</h3>
              <button
                onClick={() => setSelectedSources(null)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="p-4 overflow-y-auto max-h-[60vh]">
              <div className="space-y-3">
                {selectedSources.map((source, index) => (
                  <a
                    key={index}
                    href={source}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block p-3 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="text-blue-300 font-medium truncate">
                          {source.includes('http') ? new URL(source).hostname : source}
                        </div>
                        <div className="text-gray-400 text-sm truncate mt-1">
                          {source}
                        </div>
                      </div>
                      <ExternalLink className="h-4 w-4 text-blue-400 ml-2 flex-shrink-0" />
                    </div>
                  </a>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </MapContainer>
  );
};

export default IncidentMap;

