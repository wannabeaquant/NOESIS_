import React, { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Progress } from "../components/ui/progress";
import { Alert, AlertDescription } from "../components/ui/alert";
import {
  TrendingUp,
  AlertTriangle,
  Clock,
  MapPin,
  Activity,
  Satellite,
  DollarSign,
  Wifi,
} from "lucide-react";
import apiService from "../services/api";

const PredictiveDashboard = () => {
  const [predictions, setPredictions] = useState([]);
  const [riskAssessment, setRiskAssessment] = useState(null);
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPredictiveData();
    const interval = setInterval(fetchPredictiveData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchPredictiveData = async () => {
    try {
      const [predictionsData, riskData, dashboardData] = await Promise.all([
        apiService.fetchPredictions(),
        apiService.fetchRiskAssessment(),
        apiService.fetchPredictiveDashboard(),
      ]);

      setPredictions(predictionsData || []);
      setRiskAssessment(riskData);
      setDashboardData(dashboardData);
    } catch (error) {
      console.error("Error fetching predictive data:", error);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level) => {
    switch (level) {
      case "high":
        return "bg-red-500";
      case "medium":
        return "bg-yellow-500";
      case "low":
        return "bg-green-500";
      default:
        return "bg-gray-500";
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case "high":
        return "text-red-600";
      case "medium":
        return "text-yellow-600";
      case "low":
        return "text-green-600";
      default:
        return "text-gray-600";
    }
  };

  const formatTimeToIncident = (timeStr) => {
    try {
      const time = new Date(timeStr);
      const now = new Date();
      const diffHours = Math.round((time - now) / (1000 * 60 * 60));
      return `${diffHours} hours`;
    } catch {
      return "Unknown";
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
          NOESIS Predictive Intelligence Dashboard
        </h1>
        <p className="text-lg font-medium text-blue-600">
          AI-powered forecasting and risk assessment
        </p>
      </div>

      {/* Risk Assessment Overview */}
      {riskAssessment && (
        <Card className="border-l-4 border-l-blue-500">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Overall Risk Assessment
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div
                  className={`inline-flex items-center px-3 py-1 rounded-full text-white ${getRiskColor(
                    riskAssessment.overall_risk_level
                  )}`}
                >
                  {riskAssessment.overall_risk_level.toUpperCase()}
                </div>
                <p className="text-sm text-gray-600 mt-1">Risk Level</p>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {(riskAssessment.risk_score * 100).toFixed(1)}%
                </div>
                <p className="text-sm text-gray-600">Risk Score</p>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {riskAssessment.active_predictions}
                </div>
                <p className="text-sm text-gray-600">Active Predictions</p>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">
                  {riskAssessment.high_confidence_predictions}
                </div>
                <p className="text-sm text-gray-600">High Confidence</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Data Sources Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5 text-blue-500" />
              <div>
                <p className="text-sm font-medium">Social Media</p>
                <p className="text-xs text-gray-500">Twitter, Reddit</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Satellite className="h-5 w-5 text-green-500" />
              <div>
                <p className="text-sm font-medium">Satellite Data</p>
                <p className="text-xs text-gray-500">Crowd Detection</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Wifi className="h-5 w-5 text-purple-500" />
              <div>
                <p className="text-sm font-medium">IoT Sensors</p>
                <p className="text-xs text-gray-500">Traffic, Environment</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <DollarSign className="h-5 w-5 text-yellow-500" />
              <div>
                <p className="text-sm font-medium">Financial Data</p>
                <p className="text-xs text-gray-500">Market Indicators</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Predictions */}
      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-white-900">
          Active Predictions
        </h2>
        {predictions.length === 0 ? (
          <Card>
            <CardContent className="p-6 text-center">
              <AlertTriangle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">
                No high-confidence predictions at this time
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {predictions.map((prediction, index) => (
              <Card key={index} className="border-l-4 border-l-orange-500">
                <CardHeader>
                  <div className="flex justify-between items-start">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        <MapPin className="h-4 w-4" />
                        {prediction.location}
                      </CardTitle>
                      <p className="text-sm text-gray-600">
                        Predicted in{" "}
                        {formatTimeToIncident(
                          prediction.predicted_incident_time
                        )}
                      </p>
                    </div>
                    <Badge
                      className={getSeverityColor(
                        prediction.predicted_severity
                      )}
                    >
                      {prediction.predicted_severity.toUpperCase()}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {/* Prediction Reason */}
                    <div className="text-sm text-gray-600 bg-gray-50 p-2 rounded">
                      {prediction.prediction_reason}
                    </div>

                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Confidence</span>
                        <span>{(prediction.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <Progress
                        value={prediction.confidence * 100}
                        className="h-2"
                      />
                    </div>

                    {/* Risk Factors */}
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="flex justify-between">
                        <span>Recent Incidents</span>
                        <span className="font-medium">
                          {prediction.risk_factors.recent_incidents}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>High Severity</span>
                        <span className="font-medium">
                          {prediction.risk_factors.high_severity}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Threat Level</span>
                        <span className="font-medium">
                          {prediction.risk_factors.threat_level}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>ML Confidence</span>
                        <span className="font-medium">
                          {(
                            prediction.risk_factors.ml_confidence * 100
                          ).toFixed(1)}
                          %
                        </span>
                      </div>
                    </div>

                    {/* Real-time Threat Indicators */}
                    {prediction.risk_factors.real_time_indicators &&
                      prediction.risk_factors.real_time_indicators.length >
                        0 && (
                        <div className="mt-3">
                          <h4 className="text-sm font-medium text-gray-700 mb-2">
                            Real-time Threat Indicators:
                          </h4>
                          <div className="space-y-2">
                            {prediction.risk_factors.real_time_indicators.map(
                              (indicator, idx) => (
                                <div
                                  key={idx}
                                  className="text-xs bg-orange-50 p-2 rounded border-l-2 border-orange-300"
                                >
                                  <div className="flex justify-between items-center">
                                    <span className="font-medium text-orange-800 capitalize">
                                      {indicator.source.replace("_", " ")}
                                    </span>
                                    <span
                                      className={`px-1 rounded text-xs ${
                                        indicator.trend === "increasing"
                                          ? "bg-red-100 text-red-700"
                                          : indicator.trend === "decreasing"
                                          ? "bg-green-100 text-green-700"
                                          : "bg-gray-100 text-gray-700"
                                      }`}
                                    >
                                      {indicator.trend}
                                    </span>
                                  </div>
                                  <div className="text-gray-600 mt-1">
                                    {indicator.description}
                                  </div>
                                  <div className="flex justify-between text-gray-500 mt-1">
                                    <span>
                                      Value:{" "}
                                      {(indicator.value * 100).toFixed(1)}%
                                    </span>
                                    <span>
                                      Confidence:{" "}
                                      {(indicator.confidence * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                </div>
                              )
                            )}
                          </div>
                        </div>
                      )}

                    {/* Based on Incidents */}
                    {prediction.risk_factors.based_on_incidents &&
                      prediction.risk_factors.based_on_incidents.length > 0 && (
                        <div className="mt-3">
                          <h4 className="text-sm font-medium text-gray-700 mb-2">
                            Based on these incidents:
                          </h4>
                          <div className="space-y-2">
                            {prediction.risk_factors.based_on_incidents.map(
                              (incident, idx) => (
                                <div
                                  key={idx}
                                  className="text-xs bg-blue-50 p-2 rounded border-l-2 border-blue-300"
                                >
                                  <div className="font-medium text-blue-800">
                                    {incident.title}
                                  </div>
                                  <div className="flex justify-between text-gray-600 mt-1">
                                    <span>ID: {incident.id}</span>
                                    <span
                                      className={`px-1 rounded text-xs ${
                                        incident.severity === "high"
                                          ? "bg-red-100 text-red-700"
                                          : incident.severity === "medium"
                                          ? "bg-yellow-100 text-yellow-700"
                                          : "bg-green-100 text-green-700"
                                      }`}
                                    >
                                      {incident.severity}
                                    </span>
                                    <span>
                                      {incident.sources_count} sources
                                    </span>
                                  </div>
                                </div>
                              )
                            )}
                          </div>
                        </div>
                      )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* System Status */}
      {dashboardData && (
        <Card>
          <CardHeader>
            <CardTitle>System Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {dashboardData.prediction_accuracy}
                </div>
                <p className="text-sm text-gray-600">Prediction Accuracy</p>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {dashboardData.recent_incidents}
                </div>
                <p className="text-sm text-gray-600">Recent Incidents (24h)</p>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {dashboardData.system_status}
                </div>
                <p className="text-sm text-gray-600">System Status</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default PredictiveDashboard;
