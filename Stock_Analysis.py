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
import re

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
    
    except Exception:
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
    **Predicted Prices (Next 7 Days):** {predictions}  
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
        raw_text = response.choices[0].message.content.strip()

        # Remove the "Buy, Hold, or Sell?" instructions
        cleaned_text = raw_text.replace(
            "**(Buy, Hold, or Sell? Justify using SMA, EMA, RSI, earnings, or macro factors.)**",
            ""
        )

        # Convert '**' bold markdown to <strong> HTML
        recommendation_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", cleaned_text)

        # Replace newlines with <br>
        recommendation_html = recommendation_html.replace("\n", "<br>")

        return recommendation_html
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

def forecast_next_7_days(stock_data):
    """
    Generates a 7-business-day forecast using ARIMA.
    """
    y = stock_data['Close']
    # Fit ARIMA
    model = ARIMA(y, order=(5,1,0))
    model_fit = model.fit()

    steps = 7  # next 7 business days
    forecast = model_fit.forecast(steps=steps)
    
    # Build dict: {date_string: price}
    predictions = {}
    # Start from the day *after* the last historical date
    last_date = stock_data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=steps+1, freq='B')[1:]

    for dt, pred in zip(future_dates, forecast):
        predictions[dt.strftime("%Y-%m-%d")] = round(float(pred), 2)

    return predictions

def plot_7_day_forecast(stock_ticker, predictions):
    """
    Plots ONLY the 7-day forecast data.
    """
    if not predictions:
        st.error("⚠️ No forecast data available to plot.")
        return

    # Convert the predictions dict to a DataFrame
    forecast_df = pd.DataFrame({
        "Date": list(predictions.keys()),
        "Predicted Price": list(predictions.values())
    })
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

    fig = px.line(
        forecast_df.sort_values("Date"),
        x="Date",
        y="Predicted Price",
        title=f"{stock_ticker} - 7-Day Forecast",
        labels={"Predicted Price": "Price ($)", "Date": "Date"},
        template="plotly_white",
        line_shape="spline"
    )

    st.plotly_chart(fig, use_container_width=True)

###############################################################################
#                          Streamlit App Interface                            #
###############################################################################

st.title("📈 Stock Analysis & Prediction App")

stock_ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, GOOG):").strip().upper()

# Disclaimer in smaller font
st.markdown(
    """
    <div style="font-size:0.85rem; color:#666; margin: 8px 0 20px 0;">
    <strong>Disclaimer:</strong> This application is intended <strong>solely</strong> 
    for technical demonstration and educational purposes. It is <strong>not</strong> 
    financial or investment advice. Do not make any investment decisions based on 
    this content. The creator assumes <strong>no responsibility</strong> for any 
    actions taken based on the information presented. Always consult a qualified 
    financial professional before making any investment choices.
    </div>
    """,
    unsafe_allow_html=True
)

if stock_ticker:
    st.subheader(f"Analyzing {stock_ticker}...")
    stock_data = fetch_stock_data(stock_ticker, months=12)

    if stock_data.empty:
        st.error(f"⚠️ No data found for '{stock_ticker}'. Please check the ticker symbol.")
    else:
        st.success(f"Got it! Evaluating {stock_ticker} stock...")

        # Forecast for next 7 business days
        predictions = forecast_next_7_days(stock_data)
        # Plot only those predictions
        plot_7_day_forecast(stock_ticker, predictions)

        st.subheader("Projected Closing Prices (Next 7 Days)")
        # Create the forecast table
        forecast_items = list(predictions.items())  # [(date_str, price), ...]
        forecast_df = pd.DataFrame(forecast_items, columns=["Date", "Predicted Price ($)"])

        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
        forecast_df.sort_values("Date", inplace=True)
        forecast_df.set_index("Date", inplace=True)

        # Format the price
        forecast_df["Predicted Price ($)"] = forecast_df["Predicted Price ($)"].apply(lambda x: f"${x:,.2f}")

        st.dataframe(forecast_df, use_container_width=True)

        sentiment = fetch_news_sentiment(stock_ticker)
        recommendation_html = generate_recommendation_with_openai(stock_ticker, predictions, sentiment)

        st.subheader("💡 Investment Recommendation")
        st.markdown(f"""
        <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; background-color: #f9f9f9;">
            {recommendation_html}
            <br><br>
            <strong>Market Sentiment:</strong> {sentiment}
        </div>
        """, unsafe_allow_html=True)

