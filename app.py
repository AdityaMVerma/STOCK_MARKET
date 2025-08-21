import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("📈 Stock Analysis App")

# --- Inputs ---
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY.NS)", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))
interval = st.selectbox("Interval", ["1d", "1wk", "1mo"])

# --- Technical Indicators Selection ---
indicators = st.multiselect(
    "Select Technical Indicators to Add",
    ["SMA", "EMA", "MACD", "RSI"],
)

# --- Fetch Data ---
if st.button("Fetch Data"):
    with st.spinner("Fetching data..."):
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        if not data.empty:
            st.success(f"Data loaded for {ticker}")

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

            # Drop NaN rows so indicators plot correctly
            df = df.dropna()

            # ---------------------- Plot Price Chart ----------------------
            st.subheader("📊 Price & Indicators")
            fig = go.Figure()

            # Closing price
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close", line=dict(color="black")))

            if "SMA" in indicators and "SMA_20" in df:
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode="lines", name="SMA 20", line=dict(color="blue")))

            if "EMA" in indicators and "EMA_20" in df:
                fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], mode="lines", name="EMA 20", line=dict(color="orange")))

            fig.update_layout(title=f"{ticker} Stock Price with Indicators", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

            # ---------------------- MACD Plot ----------------------
            if "MACD" in indicators and "MACD" in df:
                st.subheader("📈 Moving Average Convergence Divergence (MACD)")
                st.markdown(
                    "The **MACD** helps identify changes in momentum, showing the relationship "
                    "between two moving averages (12-day EMA and 26-day EMA). A Signal line (9-day EMA of MACD) "
                    "is used to spot buy/sell opportunities."
                )
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="purple")))
                fig_macd.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal Line", line=dict(color="green")))
                st.plotly_chart(fig_macd, use_container_width=True)

            # ---------------------- RSI Plot ----------------------
            if "RSI" in indicators and "RSI" in df:
                st.subheader("📉 Relative Strength Index (RSI)")
                st.markdown(
                    "The **RSI** measures the speed and change of price movements. "
                    "Values above 70 typically indicate an overbought condition, while values below 30 suggest oversold conditions."
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
