import streamlit as st
import yfinance as yf
import pandas as pd

st.title("📈 Stock Analysis App")

# --- Inputs ---
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY.NS)", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))
interval = st.selectbox("Interval", ["1d", "1wk", "1mo"])

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

            # Plot Line Chart (Close price)
            st.subheader("Closing Price Chart")
            st.line_chart(data["Close"])

            # Save data locally
            data.to_csv(f"Data/{ticker}.csv")

            
        else:
            st.error("No data found for this ticker/date range.")
