import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import os
from dotenv import load_dotenv
import plotly.express as px

# Import RF model functions
from Models.RF import train_random_forest, predict_random_forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



st.title("📈 Stock Analysis App")

# ====================== SIDEBAR: Navigation ======================
st.sidebar.title("Navigation")
from pathlib import Path
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

API_KEY = os.getenv("API_KEY")

# Debug line - guaranteed to show
page = st.sidebar.radio("Go to", ["Stock Data", "Random Forest Prediction", "Sentiment Analysis"])

# ====================== HELPER: Fetch News & Sentiment ======================
def get_news_sentiment(ticker: str, api_key: str, limit: int = 15):
    """
    Fetch news sentiment data from Alpha Vantage API.
    """
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    return data.get("feed", [])
# ====================== MAIN APP ======================
if page == "Stock Data":
    st.header("📈 Stock Data Viewer")
    
    # --- Inputs ---
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY.NS)", "AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"])
    
    # Technical indicators
    indicators = st.multiselect(
        "Select Technical Indicators to Add",
        ["SMA", "EMA", "MACD", "RSI"],
    )

    if st.button("Fetch Data"):
        with st.spinner("Fetching data..."):
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

            if not data.empty:
                st.success(f"Data loaded for {ticker}")

                # Flatten MultiIndex columns if needed
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] for col in data.columns]

                # Show last few rows
                st.subheader(f"Stock Data for {ticker}")
                st.dataframe(data.tail())

                # Download CSV Button
                csv = data.to_csv().encode("utf-8")
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv,
                    file_name=f"{ticker}_stock_data.csv",
                    mime="text/csv",
                )

                # ---------------------- Technical Indicators ----------------------
                df = data.copy()

                if "SMA" in indicators:
                    df["SMA_20"] = df["Close"].rolling(window=20).mean()

                if "EMA" in indicators:
                    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

                if "MACD" in indicators:
                    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
                    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
                    df["MACD"] = df["EMA_12"] - df["EMA_26"]
                    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

                if "RSI" in indicators:
                    delta = df["Close"].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(14).mean()
                    avg_loss = loss.rolling(14).mean()
                    rs = avg_gain / avg_loss
                    df["RSI"] = 100 - (100 / (1 + rs))

                df = df.dropna()  # Drop NaN rows for correct plotting

                # ---------------------- Closing Price with SMA & EMA ----------------------
                if "SMA" in indicators or "EMA" in indicators:
                    st.subheader("📊 Simple & Exponential Moving Averages (SMA & EMA)")
                    st.markdown(
                        """
                        **SMA (Simple Moving Average):** Averages the stock price over a period (e.g., 20 days) to smooth out short-term fluctuations.  
                        **EMA (Exponential Moving Average):** Similar to SMA but gives more weight to recent prices, making it more responsive to recent changes.
                        """
                    )
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close", line=dict(color="blue")))

                    if "SMA" in indicators and "SMA_20" in df:
                        fig_price.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode="lines", name="SMA 20", line=dict(color="orange")))

                    if "EMA" in indicators and "EMA_20" in df:
                        fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], mode="lines", name="EMA 20", line=dict(color="green")))

                    fig_price.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Price",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_price, use_container_width=True)

                # ---------------------- MACD Plot ----------------------
                if "MACD" in indicators and "MACD" in df:
                    st.subheader("📈 Moving Average Convergence Divergence (MACD)")
                    st.markdown(
                        """
                        **MACD** shows the difference between 12-day EMA and 26-day EMA.  
                        **Signal line** (9-day EMA of MACD) is used to identify potential buy/sell points.
                        """
                    )
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="purple")))
                    fig_macd.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal Line", line=dict(color="green")))
                    st.plotly_chart(fig_macd, use_container_width=True)

                # ---------------------- RSI Plot ----------------------
                if "RSI" in indicators and "RSI" in df:
                    st.subheader("📉 Relative Strength Index (RSI)")
                    st.markdown(
                        """
                        **RSI** measures price speed & change.  
                        - Above 70: overbought (may drop)  
                        - Below 30: oversold (may rise)
                        """
                    )
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="blue")))
                    fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
                    st.plotly_chart(fig_rsi, use_container_width=True)

                # Save data locally
                df.to_csv(f"Data/{ticker}.csv")

            else:
                st.error("No data found for this ticker/date range.")


elif page == "Random Forest Prediction":
    st.header("🔮 Random Forest Stock Prediction")
    
    ticker_pred = st.text_input("Enter Stock Ticker for Prediction", "AAPL")
    
    if st.button("Run Prediction"):
        with st.spinner("Training Random Forest model..."):
            hist_data = yf.download(
                ticker_pred,
                start=pd.to_datetime("today") - pd.DateOffset(years=2),
                end=pd.to_datetime("today"),
                interval="1d"
            )
            if not hist_data.empty:
                if isinstance(hist_data.columns, pd.MultiIndex):
                    hist_data.columns = [col[0] for col in hist_data.columns]

                hist_data["Return"] = hist_data["Close"].pct_change()
                hist_data = hist_data.dropna()

                X = hist_data[["Open", "High", "Low", "Volume", "Return"]]
                y = hist_data["Close"]

                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                model = train_random_forest(X_train, y_train)
                results = predict_random_forest(model, X_test, y_test)

                st.success("✅ Model trained successfully!")

                # Compute RMSE for extra info
                rmse = np.sqrt(mean_squared_error(y_test, results["predictions"]))

                st.subheader("📊 Model Performance Metrics")
                st.write(f" **MAE (Mean Absolute Error):** Average of absolute differences between predicted & actual values.** " )
                st.write(f"**MAE:** {results['mae']:.2f}")
                st.write(f"**MSE (Mean Squared Error):** Average squared differences; penalizes larger errors. ** " )
                st.write(f"**MSE:** {results['mse']:.2f}")
                st.write(f" **RMSE (Root Mean Squared Error):** Square root of MSE; same units as target, easier to interpret. ** " )
                st.write(f"**RMSE:** {rmse:.2f}")
                st.write(f" ****R² (R-squared):** Proportion of variance in the dependent variable explained by the model.Higher R² is better; lower errors are better.** " )
                st.write(f"**R²:** {results['r2']:.2f}")

                st.subheader("📈 Last Predictions vs Actual")
                compare_df = pd.DataFrame({
                    "Actual": y_test.tail(10).values,
                    "Predicted": results["predictions"][-10:]
                }, index=y_test.tail(10).index)
                st.dataframe(compare_df)
            else:
                st.error("Ticker Not found / Not enough historical data for training.")

        # ---------------------- Future 2 Weeks Prediction using 5-day lags ----------------------
        st.subheader("🔮 Future 2 Weeks Predictions (5-day lag)")

        # Prepare 5-day lagged features
        lag_days = 5
        hist_data = hist_data.copy()
        for i in range(1, lag_days + 1):
            hist_data[f'lag_{i}'] = hist_data['Close'].shift(i)

        hist_data = hist_data.dropna()

        # Features and target
        feature_cols = [f'lag_{i}' for i in range(1, lag_days + 1)]
        X = hist_data[feature_cols]
        y = hist_data['Close']

        # Train Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Generate future predictions
        last_known = hist_data['Close'].iloc[-lag_days:].tolist()
        future_preds = []
        future_dates = pd.bdate_range(start=hist_data.index[-1] + pd.Timedelta(days=1), periods=14)

        for _ in future_dates:
            X_next = np.array(last_known[-lag_days:]).reshape(1, -1)
            pred = model.predict(X_next)[0]
            future_preds.append(pred)
            last_known.append(pred)  # Append prediction to use in next iteration

        # Scale chart for better visualization
        future_min = min(future_preds)
        future_max = max(future_preds)
        buffer = (future_max - future_min) * 0.1  # 10% buffer
        y_range = [future_min - buffer, future_max + buffer]

        # Create Plotly line chart
        import plotly.graph_objects as go

        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(
            x=future_dates,
            y=future_preds,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            name='Predicted Close'
        ))
        fig_future.update_layout(
            title="Predicted Close Prices for Next 2 Weeks",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis=dict(range=y_range),
            template="plotly_white"
        )

        st.plotly_chart(fig_future, use_container_width=True)
    else:
        st.error("Ticker Not Found/ Not enough historical data available")


elif page == "Sentiment Analysis":
    st.header("📰 Market News & Sentiment")
    ticker_sent = st.text_input("Enter Stock Ticker for Sentiment Analysis", "AAPL")
    st.markdown("""
                    **ℹ️ Sentiment Score Interpretation**  
                    - ≤ -0.35 → Bearish  
                    - -0.35 to -0.15 → Somewhat Bearish  
                    - -0.15 to 0.15 → Neutral  
                    - 0.15 to 0.35 → Somewhat Bullish  
                    - ≥ 0.35 → Bullish  
                    """)

    if st.button("Fetch Sentiment"):
        if not API_KEY:
            st.error("API key not found. Please set API_KEY in your .env file.")
        else:
            with st.spinner("Fetching sentiment data..."):
                # ✅ Fetch 50 articles
                news_feed = get_news_sentiment(ticker_sent, API_KEY, limit=50)

                if news_feed:
                    sentiments = []

                    # ✅ Show only first 15 articles
                    for article in news_feed[:15]:
                        st.markdown(f"### [{article['title']}]({article['url']})")
                        if "banner_image" in article and article["banner_image"]:
                            st.image(article["banner_image"], width=400)
                        st.caption(f"Source: {article['source']} | Published: {article['time_published']}")
                        st.write(article['summary'])

                        # ✅ Show overall article sentiment
                        st.write(f"**Overall Sentiment:** {article['overall_sentiment_label']} "
                                 f"({article['overall_sentiment_score']:.2f})")

                        # ✅ Extract ticker-specific sentiment
                        ticker_sentiment = None
                        for ts in article.get("ticker_sentiment", []):
                            if ts["ticker"].upper() == ticker_sent.upper():
                                ticker_sentiment = ts
                                break

                        if ticker_sentiment:
                            st.write(
                                f"**{ticker_sent} Sentiment:** {ticker_sentiment['ticker_sentiment_label']} "
                                f"({float(ticker_sentiment['ticker_sentiment_score']):.2f})"
                            )
                            sentiments.append(ticker_sentiment["ticker_sentiment_label"])
                        else:
                            st.write(f"**{ticker_sent} Sentiment:** Not available in this article")

                        st.divider()

                    # ✅ Chart for ALL 50 articles (ticker-specific sentiment only)
                    all_sentiments = []
                    for article in news_feed:
                        for ts in article.get("ticker_sentiment", []):
                            if ts["ticker"].upper() == ticker_sent.upper():
                                all_sentiments.append(ts["ticker_sentiment_label"])
                                break

                    if all_sentiments:
                        sentiment_df = pd.DataFrame(all_sentiments, columns=["Sentiment"])
                        fig = px.histogram(
                            sentiment_df,
                            x="Sentiment",
                            title=f"{ticker_sent} Sentiment Distribution (All 50 Articles)",
                            color="Sentiment"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No ticker-specific sentiment data found for {ticker_sent}.")

                else:
                    st.warning("No news data available at the moment.")
