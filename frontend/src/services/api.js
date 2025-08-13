const API_BASE_URL = 'http://localhost:8000';

class ApiService {
  async fetchIncidents(params = {}) {
    try {
      const queryParams = new URLSearchParams();
      if (params.region) queryParams.append('region', params.region);
      if (params.severity) queryParams.append('severity', params.severity);
      if (params.limit) queryParams.append('limit', params.limit);
      
      const response = await fetch(`${API_BASE_URL}/incidents/?${queryParams}`);
      if (!response.ok) throw new Error('Failed to fetch incidents');
      return await response.json();
    } catch (error) {
      console.error('Error fetching incidents:', error);
      return [];
    }
  }

  async fetchLatestIncidents(limit = 10) {
    try {
      const response = await fetch(`${API_BASE_URL}/incidents/latest?limit=${limit}`);
      if (!response.ok) throw new Error('Failed to fetch latest incidents');
      return await response.json();
    } catch (error) {
      console.error('Error fetching latest incidents:', error);
      return [];
    }
  }

  async fetchDashboard() {
    try {
      const response = await fetch(`${API_BASE_URL}/incidents/dashboard`);
      if (!response.ok) throw new Error('Failed to fetch dashboard data');
      return await response.json();
    } catch (error) {
      console.error('Error fetching dashboard:', error);
      return null;
    }
  }

  async triggerCollection() {
    try {
      const response = await fetch(`${API_BASE_URL}/collection/run-cycle`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error('Failed to trigger collection');
      return await response.json();
    } catch (error) {
      console.error('Error triggering collection:', error);
      return null;
    }
  }

  async subscribeToAlerts(subscriptionData) {
    try {
      const response = await fetch(`${API_BASE_URL}/alerts/subscribe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(subscriptionData),
      });
      if (!response.ok) throw new Error('Failed to subscribe to alerts');
      return await response.json();
    } catch (error) {
      console.error('Error subscribing to alerts:', error);
      return null;
    }
  }

  async getCollectionStatus() {
    try {
      const response = await fetch(`${API_BASE_URL}/collection/status`);
      if (!response.ok) throw new Error('Failed to fetch collection status');
      return await response.json();
    } catch (error) {
      console.error('Error fetching collection status:', error);
      return null;
    }
  }

  async fetchPredictions(confidenceThreshold = 0.3) {
    try {
      const response = await fetch(`${API_BASE_URL}/predictions/?confidence_threshold=${confidenceThreshold}`);
      if (!response.ok) throw new Error('Failed to fetch predictions');
      return await response.json();
    } catch (error) {
      console.error('Error fetching predictions:', error);
      return [];
    }
  }

  async fetchRiskAssessment() {
    try {
      const response = await fetch(`${API_BASE_URL}/predictions/risk-assessment`);
      if (!response.ok) throw new Error('Failed to fetch risk assessment');
      return await response.json();
    } catch (error) {
      console.error('Error fetching risk assessment:', error);
      return null;
    }
  }

  async fetchPredictiveDashboard() {
    try {
      const response = await fetch(`${API_BASE_URL}/predictions/dashboard`);
      if (!response.ok) throw new Error('Failed to fetch predictive dashboard');
      return await response.json();
    } catch (error) {
      console.error('Error fetching predictive dashboard:', error);
      return null;
    }
  }
}

export default new ApiService(); 