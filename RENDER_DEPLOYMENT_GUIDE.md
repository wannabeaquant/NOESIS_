# NOESIS Backend Deployment Guide for Render

This guide will walk you through deploying the NOESIS backend on Render step by step.

## Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Your NOESIS project should be in a GitHub repository
3. **API Keys** (Optional): If you want to use external services like Twitter, Reddit, etc.

## Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Ensure your repository structure is correct**:
   ```
   NOESIS/
   ├── backend/
   │   ├── main.py
   │   ├── requirements.txt
   │   ├── render.yaml
   │   ├── runtime.txt
   │   ├── start_render.py
   │   ├── create_db.py
   │   ├── app/
   │   └── ...
   ```

2. **Commit and push all changes to GitHub**:
   ```bash
   git add .
   git commit -m "Add Render deployment configuration"
   git push origin main
   ```

### Step 2: Create a New Web Service on Render

1. **Log into Render Dashboard**
   - Go to [dashboard.render.com](https://dashboard.render.com)
   - Sign in with your account

2. **Create New Web Service**
   - Click "New +" button
   - Select "Web Service"
   - Connect your GitHub repository if not already connected

3. **Configure the Service**
   - **Name**: `noesis-backend` (or your preferred name)
   - **Environment**: `Python 3`
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: `NOESIS/backend` (important!)
   - **Build Command**: `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
   - **Start Command**: `python start_render.py`

### Step 3: Configure Environment Variables

In the Render dashboard, go to your service's "Environment" tab and add these variables:

#### Required Variables:
```
PYTHON_VERSION=3.10.0
PORT=8000
DATABASE_URL=sqlite:///./noesis.db
HOST=0.0.0.0
RELOAD=false
DEBUG=false
ENVIRONMENT=production
LOG_LEVEL=INFO
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

#### Optional Variables (for full functionality):
```
# Twitter API (Optional)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

# Reddit API (Optional)
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=NOESIS_Bot/1.0

# News API (Optional)
NEWS_API_KEY=your_news_api_key_here

# Weather API (Optional)
WEATHER_API_KEY=your_weather_api_key_here

# Telegram API (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# Email Configuration (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password_here
```

### Step 4: Deploy

1. **Click "Create Web Service"**
   - Render will automatically start building and deploying your application
   - This process may take 5-10 minutes for the first deployment

2. **Monitor the Build Logs**
   - Watch the build logs for any errors
   - The build will install dependencies and download spaCy models

### Step 5: Verify Deployment

1. **Check Service Status**
   - Your service should show "Live" status
   - Note the provided URL (e.g., `https://noesis-backend.onrender.com`)

2. **Test the API**
   - Visit your service URL to see the welcome message
   - Visit `/docs` for the interactive API documentation
   - Visit `/health` for the health check endpoint

3. **Test API Endpoints**
   ```bash
   # Test root endpoint
   curl https://your-service-url.onrender.com/
   
   # Test health endpoint
   curl https://your-service-url.onrender.com/health
   ```

## Important Notes

### Database
- The application uses SQLite by default, which is stored in the container's filesystem
- **Important**: SQLite data will be lost when the container restarts
- For production, consider using a PostgreSQL database service on Render

### Free Tier Limitations
- Render's free tier has limitations:
  - Services sleep after 15 minutes of inactivity
  - Limited bandwidth and build minutes
  - Consider upgrading for production use

### Scaling
- The starter plan is sufficient for development and small-scale use
- Upgrade to higher plans for better performance and reliability

## Troubleshooting

### Common Issues

1. **Build Fails**
   - Check the build logs for specific error messages
   - Ensure all dependencies are in `requirements.txt`
   - Verify Python version compatibility

2. **Service Won't Start**
   - Check the start command in render.yaml
   - Verify the main.py file exists and is correct
   - Check environment variables are set correctly

3. **Database Issues**
   - Ensure the database directory is writable
   - Check if create_db.py runs successfully

4. **Port Issues**
   - Render automatically sets the PORT environment variable
   - Your application must use this port, not hardcoded values

### Getting Help

1. **Check Render Logs**: Use the "Logs" tab in your service dashboard
2. **Review Build Logs**: Check for dependency installation issues
3. **Test Locally**: Ensure the application runs locally before deploying

## Next Steps

After successful deployment:

1. **Update Frontend**: Point your frontend to the new backend URL
2. **Set up Custom Domain** (Optional): Configure a custom domain in Render
3. **Monitor Performance**: Use Render's built-in monitoring tools
4. **Set up Alerts**: Configure notifications for service status

## API Documentation

Once deployed, visit:
- `https://your-service-url.onrender.com/docs` - Interactive API docs
- `https://your-service-url.onrender.com/redoc` - Alternative API docs

Your NOESIS backend is now live on Render! 🚀
