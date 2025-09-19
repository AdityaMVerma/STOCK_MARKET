# 📈 Stock Market App 

🔗 **Live Demo**: [https://stock-app-3xhh.onrender.com](https://stock-app-3xhh.onrender.com)  
A Streamlit-based web app for analyzing stock data and market sentiment.
- Fetches live and historical stock prices with yFinance -
- Provides technical indicators (SMA, EMA, etc.)
- Displays latest news & sentiment via Alpha Vantage API
- Fully containerized using Docker

## 🚀 Features

- 📊 Stock price visualization with moving averages (5-day, EMA/SMA)
- 📰 Market news sentiment analysis (both overall and ticker-specific)
- 🔍 Sentiment distribution charts across 50 news articles
- ⚡ Fast deployment with Docker/Render

## 📂 Project Structure

STOCK_MARKET/
│── app.py # Main Streamlit application
│── Models/
│ └── RF.py # Random Forest training/predict helpers
│── requirements.txt # Python dependencies
│── Dockerfile # Docker build file
│── .env # API keys (NOT committed)
│── Data/ # Optional local data storage (ignored by Git)
│── README.md # This file


## ⚙️ Setup

1. Clone the repo

``` git clone https://github.com/AdityaMVerma/stock-app.git
cd stock-app ```

2. Create & activate virtual environment

```   python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

3. Install dependencies

``pip install -r requirements.txt``

4. Create .env file

Add your Alpha Vantage API Key

``ALPHA_VANTAGE_API_KEY=your_api_key_here``

5. Run the app locally

`` streamlit run app.py ``

App will be available at 👉 http://localhost:8501

## 🐳 Run with Docker

1. Build the image

``docker build -t stock-app . ``

2. Run the container

`` docker run -p 8501:8501 --env-file .env stock-app ``

Then open 👉 http://localhost:8501

## 📝 .gitignore & .dockerignore

The following are ignored to keep the repo clean and secure:

__pycache__/
*.pyc
*.pyo
*.pyd
*.db
*.sqlite3
.env
.venv
.idea/
Data/

## 🌐 Deployment

You can deploy this app easily on:
### Docker Hub → Pull & run anywhere

1. Pull the image: 
This downloads the image from Docker Hub to their local machine.

`` docker pull adsdocker728/stock-app:latest ``


2. Run the container

`` docker run -p 8501:8501 --env-file .env adsdocker728/stock-app:latest ``

### Render → Run as a web service

1. Push your image to Docker Hub

`` docker push your_docker_username/stock-app:latest ``

2. Create a new Render Web Service

- Go to https://dashboard.render.com
- Click New → Web Service
- Choose Deploy an existing image from Docker Hub
-Enter your image:
your_docker_username/stock-app:latest
- Configure service

``` Name → e.g., stock-app
Region → pick closest to you(ohio if using apis)
Environment → Docker
Port → 8501 (Streamlit default)
Branch → (not needed since we’re using Docker image)
Add environment variables 
Add:ALPHA_VANTAGE_API_KEY=your_api_key
Deploy
```

3. Click Create Web Service
Render will pull the Docker image from Docker Hub and start it.
Once deployed, Render gives you a public URL like:

`` https://stock-app.onrender.com `` 
