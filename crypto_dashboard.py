import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Constants
DAYS_TO_FETCH = 30  # Increased from 7 to 30 for better technical analysis

st.set_page_config(page_title="Crypto Price Tracker", layout="centered")

# -- CSS for subtle card style and spacing --
st.markdown(
    """
    <style>
    .card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    .header-text {
        font-weight: 700;
        font-size: 48px;
        margin-bottom: 0.2rem;
        color: #111827;
        font-family: 'Inter', sans-serif;
    }
    .subheader-text {
        font-weight: 400;
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero Section
st.markdown('<div class="header-text">Crypto Price Tracker Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-text">Track and visualize cryptocurrency price trends in real-time.</div>', unsafe_allow_html=True)

# Sidebar - Crypto selection
st.sidebar.header("Select Cryptocurrency")
cryptos = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Cardano (ADA)": "cardano",
    "Solana (SOL)": "solana",
    "Ripple (XRP)": "ripple",
}
selected_name = st.sidebar.selectbox("Cryptocurrency", list(cryptos.keys()))
selected_id = cryptos[selected_name]

# Data fetching function using CoinGecko API with OHLCV data
@st.cache_data(ttl=300)
def fetch_market_data(crypto_id, days=30):
    # First get the price data
    price_url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
    response = requests.get(price_url)
    
    if response.status_code != 200:
        st.error("Failed to fetch price data from API.")
        return None
        
    data = response.json()
    prices = data.get("prices", [])
    
    if not prices:
        st.error("No price data found.")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit='ms')
    df["date_only"] = df["date"].dt.date
    
    # Get OHLC data (Open, High, Low, Close)
    # For simplicity, we'll derive OHLC from the daily data
    # In a production app, you might want to use a different endpoint or method
    df = df.set_index('date')
    
    # Resample to daily data and calculate OHLC
    df_ohlc = df['price'].resample('D').ohlc()
    df_volume = df['price'].resample('D').count()  # Using count as a proxy for volume
    
    # Combine the data
    df = pd.concat([df_ohlc, df_volume], axis=1)
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.dropna()
    df = df.reset_index()
    df['date_only'] = df['date'].dt.date
    
    # Sort by date to ensure correct order
    df = df.sort_values('date')
    
    # Calculate technical indicators
    df = add_technical_indicators(df)
    
    return df

def add_technical_indicators(df):
    # Calculate Moving Averages
    df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
    
    # Calculate RSI
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    
    # Calculate MACD
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # Calculate Bollinger Bands
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_low'] = bb.bollinger_lband()
    
    return df

def plot_candlestick_chart(df):
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # Add moving averages
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sma_20'],
        name='SMA 20',
        line=dict(color='#ff9800', width=1.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['ema_50'],
        name='EMA 50',
        line=dict(color='#2196f3', width=1.5)
    ))
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['bb_high'],
        name='BB Upper',
        line=dict(color='#9e9e9e', width=1, dash='dash'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['bb_mid'],
        name='BB Middle',
        line=dict(color='#9e9e9e', width=1, dash='dot'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['bb_low'],
        name='BB Lower',
        line=dict(color='#9e9e9e', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(158, 158, 158, 0.1)',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title='Price Chart with Technical Indicators',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        showlegend=True,
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified'
    )
    
    return fig

def plot_volume_chart(df):
    fig = go.Figure()
    
    # Add volume bars
    colors = ['#26a69a' if close > open_ else '#ef5350' 
              for close, open_ in zip(df['close'], df['open'])]
    
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.7
    ))
    
    # Update layout
    fig.update_layout(
        title='Trading Volume',
        xaxis_title='Date',
        yaxis_title='Volume',
        template='plotly_dark',
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def plot_rsi_chart(df):
    fig = go.Figure()
    
    # Add RSI line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['rsi'],
        name='RSI (14)',
        line=dict(color='#9c27b0', width=2)
    ))
    
    # Add overbought/oversold levels
    fig.add_hline(y=70, line_dash='dash', line_color='#f44336', 
                 annotation_text='Overbought (70)', annotation_position='top right')
    fig.add_hline(y=30, line_dash='dash', line_color='#4caf50',
                 annotation_text='Oversold (30)', annotation_position='bottom right')
    
    # Update layout
    fig.update_layout(
        title='Relative Strength Index (RSI)',
        xaxis_title='Date',
        yaxis_title='RSI',
        template='plotly_dark',
        showlegend=True,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        yaxis_range=[0, 100]
    )
    
    return fig

def plot_macd_chart(df):
    fig = go.Figure()
    
    # MACD Line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['macd'],
        name='MACD',
        line=dict(color='#2196f3', width=2)
    ))
    
    # Signal Line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['macd_signal'],
        name='Signal',
        line=dict(color='#ff9800', width=2)
    ))
    
    # Histogram
    colors = ['#4caf50' if val >= 0 else '#f44336' for val in df['macd_hist']]
    
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['macd_hist'],
        name='Histogram',
        marker_color=colors,
        opacity=0.6,
        showlegend=False
    ))
    
    # Zero line
    fig.add_hline(y=0, line_width=1, line_dash='dash', line_color='#9e9e9e')
    
    # Update layout
    fig.update_layout(
        title='MACD (Moving Average Convergence Divergence)',
        xaxis_title='Date',
        yaxis_title='MACD',
        template='plotly_dark',
        showlegend=True,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

# Add time period selector in sidebar
st.sidebar.header("Time Period")
time_period = st.sidebar.selectbox("Select Time Period", 
                                 ["7 days", "14 days", "30 days", "60 days", "90 days"],
                                 index=2)  # Default to 30 days

days = int(time_period.split()[0])  # Extract number of days

# Fetch market data with loading spinner
with st.spinner(f"Loading {selected_name} market data for {time_period}..."):
    df = fetch_market_data(selected_id, days)

if df is not None and not df.empty:
    # Display Main Card with price and stats
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Latest price and basic stats
    latest_price = df.iloc[-1]["close"]
    first_price = df.iloc[0]["close"]
    pct_change = ((latest_price - first_price) / first_price) * 100
    high_price = df["high"].max()
    low_price = df["low"].min()
    volume_24h = df["volume"].iloc[-1]
    rsi = df["rsi"].iloc[-1]
    
    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Price (USD)", f"${latest_price:,.2f}")
    with col2:
        st.metric(f"{days}-day Change", f"{pct_change:+.2f}%", 
                 delta=f"${(latest_price - first_price):,.2f}")
    with col3:
        st.metric("24h Volume", f"${volume_24h:,.0f}")
    with col4:
        st.metric("RSI (14)", f"{rsi:.2f}", 
                 delta=None if not 30 <= rsi <= 70 else ("Oversold" if rsi < 30 else "Overbought"),
                 delta_color="off")
    
    # Tabs for different chart types
    tab1, tab2 = st.tabs(["📈 Price & Indicators", "📊 Technical Analysis"])
    
    with tab1:
        # Candlestick chart with volume
        st.plotly_chart(plot_candlestick_chart(df), use_container_width=True)
        
    with tab2:
        # Technical indicators in separate charts
        st.plotly_chart(plot_volume_chart(df), use_container_width=True)
        st.plotly_chart(plot_rsi_chart(df), use_container_width=True)
        st.plotly_chart(plot_macd_chart(df), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a collapsible section with raw data
    with st.expander("View Raw Data"):
        st.dataframe(df.drop(columns=['date_only']).set_index('date').sort_index(ascending=False), 
                    use_container_width=True)
else:
    st.warning("No data available to display.")

# Footer
st.markdown(
    """
    <div style="color:#6b7280; font-size:14px; padding-top:2rem; font-family: 'Inter', sans-serif;">
        Data source: CoinGecko API | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)

