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
    # Ensure we have enough data points
    if len(df) < 50:  # We need at least 50 points for reliable indicators
        return df
        
    try:
        # Calculate Moving Averages
        df['sma_20'] = SMAIndicator(close=df['close'], window=20, fillna=True).sma_indicator()
        df['ema_50'] = EMAIndicator(close=df['close'], window=50, fillna=True).ema_indicator()
        
        # Calculate RSI with more robust error handling
        rsi_indicator = RSIIndicator(close=df['close'], window=14, fillna=True)
        df['rsi'] = rsi_indicator.rsi()
        
        # Calculate MACD with standard parameters (12, 26, 9)
        macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Calculate Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2, fillna=True)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_low'] = bb.bollinger_lband()
        
        # Add additional technical indicators
        
        # 1. RSI Divergence
        df['rsi_peak'] = df['rsi'].rolling(5, center=True).max() == df['rsi']
        df['rsi_trough'] = df['rsi'].rolling(5, center=True).min() == df['rsi']
        
        # 2. MACD Signal Crossovers
        df['macd_above_signal'] = df['macd'] > df['macd_signal']
        df['macd_crossover'] = df['macd_above_signal'].ne(df['macd_above_signal'].shift())
        
        # 3. RSI with smoothed signal line
        df['rsi_signal'] = df['rsi'].rolling(window=9).mean()
        
        # 4. Volume Weighted Moving Average (VWMA)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # 5. Average True Range (ATR) for volatility
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # 6. Stochastic Oscillator
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # 7. On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # 8. Price Rate of Change (ROC)
        df['roc'] = df['close'].pct_change(periods=9) * 100
        
        # Clean up any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return df
    
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
        line=dict(color='#9c27b0', width=2),
        hovertemplate='Date: %{x}<br>RSI: %{y:.2f}<extra></extra>'
    ))
    
    # Add RSI signal line if it exists
    if 'rsi_signal' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['rsi_signal'],
            name='Signal (9)',
            line=dict(color='#ff9800', width=1.5, dash='dash'),
            hovertemplate='Signal: %{y:.2f}<extra></extra>'
        ))
    
    # Add overbought/oversold levels with colored background
    fig.add_hrect(y0=70, y1=100, line_width=0, 
                 fillcolor='rgba(244, 67, 54, 0.1)', 
                 annotation_text='Overbought', 
                 annotation_position='top right')
    
    fig.add_hrect(y0=0, y1=30, line_width=0, 
                 fillcolor='rgba(76, 175, 80, 0.1)',
                 annotation_text='Oversold',
                 annotation_position='bottom right')
    
    # Add center line
    fig.add_hline(y=50, line_dash='dot', line_color='rgba(255, 255, 255, 0.5)',
                 annotation_text='Neutral (50)', annotation_position='top left')
    
    # Add markers for divergences if they exist
    if 'rsi_peak' in df.columns:
        peaks = df[df['rsi_peak'] & (df['rsi'] > 70)]
        troughs = df[df['rsi_trough'] & (df['rsi'] < 30)]
        
        fig.add_trace(go.Scatter(
            x=peaks['date'],
            y=peaks['rsi'],
            mode='markers',
            marker=dict(color='#f44336', size=8, symbol='triangle-down'),
            name='Bearish Divergence',
            hovertemplate='Potential Bearish Divergence<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=troughs['date'],
            y=troughs['rsi'],
            mode='markers',
            marker=dict(color='#4caf50', size=8, symbol='triangle-up'),
            name='Bullish Divergence',
            hovertemplate='Potential Bullish Divergence<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Relative Strength Index (RSI)</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis_title='Date',
        yaxis_title='RSI',
        template='plotly_dark',
        showlegend=True,
        height=350,
        margin=dict(l=50, r=50, t=60, b=50),
        yaxis=dict(range=[0, 100], fixedrange=True),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    # Add custom hover info
    fig.update_traces(
        hovertemplate='<b>%{y:.2f}</b>',
        selector=dict(type='scatter')
    )
    
    return fig

def plot_macd_chart(df):
    fig = go.Figure()
    
    # Calculate histogram colors based on direction and momentum
    colors = []
    for i in range(len(df)):
        if i == 0:
            colors.append('#4caf50' if df['macd_hist'].iloc[i] >= 0 else '#f44336')
        else:
            if df['macd_hist'].iloc[i] >= 0:
                # Green for positive histogram
                if df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1]:
                    colors.append('#2e7d32')  # Darker green for increasing
                else:
                    colors.append('#4caf50')  # Lighter green for decreasing
            else:
                # Red for negative histogram
                if df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1]:
                    colors.append('#c62828')  # Darker red for decreasing
                else:
                    colors.append('#f44336')  # Lighter red for increasing
    
    # Histogram (added first to be in the background)
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['macd_hist'],
        name='Histogram',
        marker_color=colors,
        opacity=0.7,
        showlegend=False,
        hovertemplate='Hist: %{y:.4f}<extra></extra>'
    ))
    
    # MACD Line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['macd'],
        name='MACD (12,26,9)',
        line=dict(color='#2196f3', width=2),
        hovertemplate='MACD: %{y:.4f}<extra></extra>'
    ))
    
    # Signal Line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['macd_signal'],
        name='Signal Line',
        line=dict(color='#ff9800', width=1.5, dash='dash'),
        hovertemplate='Signal: %{y:.4f}<extra></extra>'
    ))
    
    # Zero line with custom styling
    fig.add_hline(
        y=0, 
        line=dict(width=1, dash='dash', color='#9e9e9e'),
        annotation_text='Zero Line',
        annotation_position='top right'
    )
    
    # Add markers for crossovers if they exist
    if 'macd_crossover' in df.columns and 'macd_above_signal' in df.columns:
        crossovers = df[df['macd_crossover'] & (df.index > 0)]
        
        for idx, row in crossovers.iterrows():
            if row['macd_above_signal']:
                # Bullish crossover (MACD crosses above signal)
                fig.add_vline(
                    x=row['date'],
                    line_width=1,
                    line_dash='dot',
                    line_color='#4caf50',
                    opacity=0.7,
                    annotation_text='Bullish',
                    annotation_position='top left'
                )
            else:
                # Bearish crossover (MACD crosses below signal)
                fig.add_vline(
                    x=row['date'],
                    line_width=1,
                    line_dash='dot',
                    line_color='#f44336',
                    opacity=0.7,
                    annotation_text='Bearish',
                    annotation_position='bottom left'
                )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>MACD (12, 26, 9)</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_dark',
        showlegend=True,
        height=400,
        margin=dict(l=50, r=50, t=60, b=50),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    # Add custom hover info
    fig.update_traces(
        hovertemplate='<b>%{y:.4f}</b>',
        selector=dict(type='scatter')
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
    
    try:
        # Latest price and basic stats
        latest_price = df.iloc[-1]["close"]
        first_price = df.iloc[0]["close"]
        pct_change = ((latest_price - first_price) / first_price) * 100
        high_price = df["high"].max()
        low_price = df["low"].min()
        volume_24h = df["volume"].iloc[-1]
        
        # Initialize RSI with None
        rsi_value = None
        rsi_text = "N/A"
        rsi_delta = None
        rsi_delta_color = "off"
        
        # Check if RSI is available
        if "rsi" in df.columns and not pd.isna(df["rsi"].iloc[-1]):
            rsi_value = df["rsi"].iloc[-1]
            rsi_text = f"{rsi_value:.2f}"
            if rsi_value < 30:
                rsi_delta = "Oversold"
            elif rsi_value > 70:
                rsi_delta = "Overbought"
        
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
            st.metric("RSI (14)", rsi_text, 
                     delta=rsi_delta,
                     delta_color=rsi_delta_color)
    except Exception as e:
        st.error(f"Error displaying price data: {str(e)}")
        st.stop()
    
    # Tabs for different chart types
    tab1, tab2 = st.tabs(["📈 Price & Indicators", "📊 Technical Analysis"])
    
    with tab1:
        try:
            # Candlestick chart with volume
            st.plotly_chart(plot_candlestick_chart(df), use_container_width=True)
        except Exception as e:
            st.warning("Could not display candlestick chart. Showing line chart instead.")
            st.line_chart(df.set_index('date')['close'])
        
    with tab2:
        try:
            # Technical indicators in separate charts
            if 'volume' in df.columns and len(df) > 0:
                st.plotly_chart(plot_volume_chart(df), use_container_width=True)
            
            if 'rsi' in df.columns and len(df) > 0:
                st.plotly_chart(plot_rsi_chart(df), use_container_width=True)
            
            if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_hist']):
                st.plotly_chart(plot_macd_chart(df), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying technical indicators: {str(e)}")
            st.warning("Some technical indicators could not be displayed. Please try a different time period or cryptocurrency.")
    
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

