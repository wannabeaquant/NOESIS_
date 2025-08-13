# NOESIS - Real-Time OSINT Civil Unrest Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)

NOESIS is a real-time OSINT (Open Source Intelligence) system designed to detect and monitor civil unrest events globally. It combines advanced NLP processing, multi-source data collection, and interactive visualization to provide comprehensive situational awareness.

## 🌟 Features

### 🔍 **Real-time Monitoring**
- **Multi-source Data Collection**: Twitter, Reddit, News APIs, Telegram, Weather
- **Live Incident Detection**: Continuous monitoring of social media and news sources
- **Geographic Intelligence**: Automatic location detection and mapping
- **Severity Assessment**: AI-powered risk evaluation and classification

### 🧠 **Advanced AI Processing**
- **NLP Pipeline**: Sentiment analysis, protest detection, and entity recognition
- **Verification System**: Multi-source verification for incident accuracy
- **Clustering Algorithm**: Groups related incidents for better analysis
- **Language Support**: Multi-language processing capabilities
- **Predictive Analytics**: ML-based risk assessment and escalation prediction

### 📊 **Interactive Dashboard**
- **Real-time Map**: Global incident visualization with interactive markers
- **Source Verification**: Direct links to original sources for fact-checking
- **Filtering System**: Filter by severity, status, and location
- **Live Updates**: Automatic refresh and real-time data streaming
- **Predictive Dashboard**: Risk scores and escalation probabilities

### 🔔 **Alert System**
- **Email Notifications**: Configurable alert delivery
- **Telegram Bot**: Real-time messaging for critical incidents
- **Customizable Thresholds**: Set alert conditions based on severity and location

## 🏗️ Architecture

```
NOESIS/
├── backend/                 # FastAPI Backend
│   ├── app/
│   │   ├── api/            # REST API endpoints
│   │   ├── collectors/     # Data collection modules
│   │   ├── services/       # Core business logic & ML models
│   │   ├── models/         # Database models
│   │   ├── alerts/         # Notification system
│   │   └── utils/          # Utility functions
│   ├── data/               # Configuration and dictionaries
│   └── tests/              # Backend tests
├── frontend/               # React Frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API integration
│   │   └── hooks/          # Custom React hooks
│   └── public/             # Static assets
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Node.js 16+**
- **Git**

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NOESIS
   ```

2. **Set up Python environment**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Configure environment variables**
   Create a `.env` file in the backend directory:
   ```env
   # Database (SQLite - no setup required)
   DATABASE_URL=sqlite:///./noesis.db

   # Twitter API v2 (optional)
   TWITTER_BEARER_TOKEN=your_twitter_bearer_token

   # Reddit API (optional)
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   REDDIT_USER_AGENT=NOESIS_Bot/1.0

   # News API (optional)
   NEWS_API_KEY=your_news_api_key

   # Weather API (optional)
   WEATHER_API_KEY=your_weather_api_key

   # Telegram API (optional - for alerts)
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token

   # Email Configuration (optional)
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=your_email@gmail.com
   SMTP_PASSWORD=your_app_password
   ```

5. **Initialize database**
   ```bash
   python create_db.py
   ```

6. **Start the backend server**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
   
   The API will be available at `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`
   - Interactive API: `http://localhost:8000/redoc`

### Frontend Setup

1. **Install Node.js dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Start the development server**
   ```bash
   npm run dev
   ```
   
   The frontend will be available at `http://localhost:5173`

## 📖 Usage

### Dashboard Features

1. **Global Map View**
   - Interactive map showing all detected incidents
   - Color-coded markers by severity (Red=High, Yellow=Medium, Green=Low)
   - Click markers to view incident details and sources

2. **Incident List**
   - Toggle between "Recent Incidents" and "All Incidents"
   - Expandable incident cards with detailed information
   - Real-time source verification links

3. **Predictive Dashboard**
   - Risk scores for different regions
   - Escalation probability indicators
   - Historical trend analysis

4. **Filtering System**
   - Filter by severity level (High/Medium/Low)
   - Filter by verification status (Verified/Pending)
   - Geographic filtering capabilities

### API Endpoints

#### Incidents
- `GET /incidents/` - Get all incidents with filtering
- `GET /incidents/latest` - Get recent incidents
- `GET /incidents/dashboard` - Get dashboard statistics
- `GET /incidents/{incident_id}` - Get specific incident

#### Predictions
- `GET /predictions/dashboard` - Get predictive analytics data
- `GET /predictions/risk-scores` - Get current risk scores

#### Collection
- `POST /collection/trigger` - Trigger data collection cycle
- `GET /collection/status` - Get collection status

#### Alerts
- `POST /alerts/subscribe` - Subscribe to alerts
- `GET /alerts/subscribers` - Get alert subscribers

#### Moderation
- `POST /moderate/flag` - Flag false positive
- `POST /moderate/confirm` - Confirm incident
- `POST /moderate/merge` - Merge duplicate incidents

## 🔧 Configuration

### Data Sources

The system supports multiple data sources:

- **Twitter**: Real-time tweet monitoring (requires API key)
- **Reddit**: Subreddit monitoring for protest discussions
- **News APIs**: RSS feeds and news article monitoring
- **Telegram**: Channel monitoring for real-time updates
- **Weather**: Weather conditions that might affect protests

### Environment Variables

All configuration is done through environment variables in the `.env` file. The system will work with minimal configuration using only free APIs and services.

## 🧪 Testing

### Backend Tests
```bash
cd backend
python -m pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Manual Testing
```bash
# Test data collection
cd backend
python quick_refresh.py

# Test full pipeline
python refresh_data.py --once
```

## 📊 Data Flow

1. **Collection**: Data collectors gather information from multiple sources
2. **Processing**: NLP pipeline analyzes text for protest indicators
3. **Verification**: Multi-source verification confirms incident accuracy
4. **Prediction**: ML models assess risk and escalation probability
5. **Storage**: Incidents stored in database with metadata
6. **Visualization**: Frontend displays data on interactive map
7. **Alerts**: Notification system sends alerts for critical incidents

## 🔒 Security Features

- **Rate Limiting**: API rate limiting to prevent abuse
- **Input Validation**: Comprehensive input sanitization
- **CORS Configuration**: Proper cross-origin resource sharing
- **Error Handling**: Graceful error handling and logging

## 🚀 Deployment

### Production Setup

1. **Backend Deployment**
   ```bash
   cd backend
   pip install -r requirements.txt
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Frontend Build**
   ```bash
   cd frontend
   npm run build
   ```

3. **Environment Configuration**
   - Set production environment variables
   - Configure reverse proxy (nginx)
   - Set up SSL certificates

### Docker Deployment (Optional)

```bash
# Build and run with Docker Compose
docker-compose up -d
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use TypeScript for frontend components
- Add tests for new features
- Update documentation for API changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI** for the robust backend framework
- **React** for the interactive frontend
- **Leaflet** for the mapping functionality
- **spaCy** for natural language processing
- **OpenStreetMap** for map data
- **Transformers** for advanced NLP capabilities

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the API documentation at `/docs`
- Review the code comments for implementation details

## 🔄 Data Refresh

The system includes automated data refresh capabilities:

```bash
# Run once
python refresh_data.py --once

# Run scheduled (every 30 minutes)
python refresh_data.py --schedule
```

---

**NOESIS** - Empowering situational awareness through intelligent OSINT analysis. 