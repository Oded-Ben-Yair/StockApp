import os
import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
from statsmodels.tsa.arima.model import ARIMA
import requests

###############################################################################
#                           OpenAI Client Setup                               #
###############################################################################

# Explicitly load the API key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("\u26a0\ufe0f OpenAI API key not found. Ensure it is set in your environment.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

@st.cache_data
def fetch_news_sentiment(stock_ticker):
    """
    Fetches recent financial news headlines for the given stock ticker
    and evaluates if they're generally positive, negative, or neutral.
    This is a simple heuristic approach and does not represent a
    detailed sentiment analysis.
    """
    try:
        url = f"https://newsapi.org/v2/everything?q={stock_ticker}&apiKey=YOUR_NEWS_API_KEY"
        response = requests.get(url).json()
        articles = response.get("articles", [])
        
        sentiment_scores = []
        for article in articles[:5]:
            title_lower = article["title"].lower()
            if "market" in title_lower:
                sentiment_scores.append(1)
            elif "drop" in title_lower:
                sentiment_scores.append(-1)
            else:
                sentiment_scores.append(0)
        
        sentiment_score = sum(sentiment_scores) / max(len(sentiment_scores), 1)
        return "📈 Bullish" if sentiment_score > 0 else "📉 Bearish" if sentiment_score < 0 else "⚖️ Neutral"
    
    except Exception as e:
        return "🔴 Unable to fetch sentiment data."

def generate_recommendation_with_openai(stock_ticker, predictions, sentiment):
    """
    Uses the OpenAI API to provide a short, structured, and easy-to-understand 
    investment recommendation. Note that this is strictly for demonstration 
    and educational purposes, not a real financial endorsement.
    """
    prompt = f"""
    You are a professional financial analyst. Provide a **clear, structured, and actionable** 
    investment recommendation in plain language so most people can understand.

    **Stock:** {stock_ticker}  
    **Predicted Prices (Next 4 Weeks):** {predictions}  
    **Market Sentiment:** {sentiment}

    ---
    **Format your response EXACTLY as follows:**
    
    **(Buy, Hold, or Sell? Justify using SMA, EMA, RSI, earnings, or macro factors.)**  

    🎯 Entry & Exit  
    - **Entry Price:** Suggest a **strong support level** for buying.  
    - **Exit Price:** Suggest a **target price based on resistance levels**.  

    📉 Alternative Scenarios  
    **Bearish Case:** Key support levels to watch if price declines.  
    **Bullish Case:** Next target price if the stock rallies.  

    ⚠️ Risks  
    - **Earnings-related risks**  
    - **Macroeconomic or industry-specific concerns**  

    📊 Forecast Model  
    **(Explain what data influenced the prediction and how in clear simple words with info from: SMA, ARIMA, news, trends.)**
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional financial analyst providing structured, "
                        "easy-to-understand investment insights."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )

        # Clean up formatting
        clean_response = response.choices[0].message.content.strip().replace("\n", "<br>")

        return clean_response
    except Exception as e:
        return f"❌ Error fetching recommendation from OpenAI API: {e}"

###############################################################################
#                      Data Fetching and Visualization                        #
###############################################################################

@st.cache_data
def fetch_stock_data(stock_ticker, months=12):
    """
    Fetches the last `months` months of daily stock data (on business days) 
    for the provided `stock_ticker`. 
    """
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30 * months)
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date, progress=False)

    # Ensure the DataFrame has a frequency set (Business Days)
    stock_data = stock_data.asfreq('B')

    return stock_data

def plot_stock_data_with_predictions(stock_ticker, stock_data, predictions):
    """
    Plots actual closing prices and predicted future prices in a user-friendly format.
    """
    if stock_data.empty:
        st.error("⚠️ No stock data available to plot.")
        return

    # Convert predictions dictionary into DataFrame
    pred_dates = pd.date_range(start=stock_data.index[-1], periods=len(predictions) + 1, freq='W')[1:]
    predictions_df = pd.DataFrame({"Date": pred_dates, "Close": list(predictions.values())})

    # Combine actual and predicted data
    stock_data = stock_data.reset_index()
    stock_data = stock_data[['Date', 'Close']]
    stock_data['Type'] = 'Actual'

    predictions_df['Type'] = 'Predicted'

    combined_df = pd.concat([stock_data, predictions_df])

    # Create the line plot with Plotly Express for better readability
    fig = px.line(
        combined_df,
        x="Date",
        y="Close",
        color="Type",
        title=f"{stock_ticker} - Actual & Predicted Prices",
        labels={"Close": "Price ($)", "Date": "Date"},
        template="plotly_white",
        line_shape="spline"
    )

    st.plotly_chart(fig, use_container_width=True)

###############################################################################
#                           Forecasting Logic                                 #
###############################################################################

def forecast_next_weeks(stock_data, weeks=4):
    """
    Uses the ARIMA model (a statistical approach) to forecast the next 
    `weeks` closing prices. ARIMA uses past price data to predict future 
    price movements. These predictions are purely illustrative.
    """
    y = stock_data['Close']
    
    model = ARIMA(y, order=(5,1,0))
    model_fit = model.fit()

    predictions = {}
    forecast = model_fit.forecast(steps=weeks)

    for i, pred in enumerate(forecast):
        predictions[f"Week {i+1}"] = round(float(pred), 2)

    return predictions

###############################################################################
#                          Streamlit App Interface                            #
###############################################################################

st.title("📈 Stock Analysis & Prediction App")

stock_ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, GOOG):").strip().upper()

if stock_ticker:
    st.subheader(f"Analyzing {stock_ticker}...")
    stock_data = fetch_stock_data(stock_ticker, months=12)

    if stock_data.empty:
        st.error(f"⚠️ No data found for '{stock_ticker}'. Please check the ticker symbol.")
    else:
        st.success("✅ Data fetched successfully!")
        st.write(stock_data.tail())

        predictions = forecast_next_weeks(stock_data, weeks=4)
        plot_stock_data_with_predictions(stock_ticker, stock_data, predictions)

        st.subheader("📊 Forecast for the Next 4 Weeks")
        
        predictions_df = pd.DataFrame(list(predictions.items()), columns=["Week", "Predicted Price ($)"])
        predictions_df["Predicted Price ($)"] = predictions_df["Predicted Price ($)"].apply(lambda x: f"${x:,.2f}")

        st.data_editor(predictions_df, use_container_width=True, height=250)

        sentiment = fetch_news_sentiment(stock_ticker)

        recommendation = generate_recommendation_with_openai(stock_ticker, predictions, sentiment)
        st.subheader("💡 Investment Recommendation")

        st.markdown(f"""
        <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; background-color: #f9f9f9;">
            <b>📌 Strategy:</b><br>
            {recommendation} <br><br>
            <b>📊 Market Sentiment:</b> {sentiment} <br>
        </div>
        """, unsafe_allow_html=True)

# ------------------ Add Disclaimer Here ------------------
st.markdown("""
---
**Disclaimer:**
This application is intended **solely** for technical demonstration and educational purposes. 
It is **not** financial or investment advice. Do not make any investment decisions based on this content. 
The creator assumes **no responsibility** for any actions taken based on the information presented. 
Always consult a qualified financial professional before making any investment choices.
""")

