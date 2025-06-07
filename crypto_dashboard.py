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

# Page configuration with wide layout and dark theme
st.set_page_config(
    page_title="Crypto Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and improved UI
st.markdown("""
    <style>
    /* Main container */
    .main .block-container {
        padding: 2rem 2.5rem;
        max-width: 1800px;
    }
    
    /* Cards */
    .card {
        background: #1e1e1e;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Headers */
    .header-text {
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
        color: #f0f2f6;
        font-family: 'Inter', sans-serif;
    }
    
    .subheader-text {
        font-weight: 400;
        font-size: 1.1rem;
        color: #9aa5b9;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 0 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 20px;
        border-radius: 8px;
        align-items: center;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #2a2a2a;
    }
    
    .stTabs [aria-selected="true"] {
        background: #3a3a3a;
        color: #f0f2f6;
        font-weight: 600;
    }
    
    /* Metrics */
    .stMetric {
        background: #252525;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #4f46e5;
    }
    
    .stMetric > div > div {
        color: #f0f2f6 !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Loading spinner */
    .stSpinner > div > div {
        border-color: #4f46e5 transparent #4f46e5 transparent !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1a1a1a;
        border-right: 1px solid #333;
    }
    
    /* Tooltips */
    .stTooltip {
        font-family: 'Inter', sans-serif;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4f46e5;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4338ca;
    }
    </style>
    """,
    unsafe_allow_html=True
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

def get_interval_for_timeframe(timeframe):
    """Convert timeframe to CoinGecko API interval"""
    if timeframe == '1m':
        return 'minute', 'm1'
    elif timeframe == '5m':
        return 'minute', 'm5'
    elif timeframe == '15m':
        return 'minute', 'm15'
    elif timeframe == '30m':
        return 'minute', 'm30'
    elif timeframe == '1h':
        return 'hourly', 'h1'
    elif timeframe == '4h':
        return 'hourly', 'h4'
    else:  # 1d and others
        return 'daily', 'd1'

# Data fetching function using CoinGecko API with OHLCV data
@st.cache_data(ttl=300)
def fetch_market_data(crypto_id, days=30, timeframe='1d'):
    # Determine the API parameters based on timeframe
    interval, _ = get_interval_for_timeframe(timeframe)
    
    # For intraday data, we'll need to fetch more data and resample
    if 'm' in timeframe or 'h' in timeframe:
        # Add buffer days to ensure we have enough data points
        days = max(days * 2, 30)  # At least 30 days of data for intraday
    
    # Get OHLC data
    ohlc_url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/ohlc?vs_currency=usd&days={days}"
    response = requests.get(ohlc_url)
    
    if response.status_code != 200:
        st.error("Failed to fetch OHLC data from API.")
        return None
        
    ohlc_data = response.json()
    
    if not ohlc_data:
        st.error("No OHLC data found.")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlc_data, columns=["timestamp", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["timestamp"], unit='ms')
    
    # Get volume data
    price_url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
    vol_response = requests.get(price_url)
    
    if vol_response.status_code == 200:
        vol_data = vol_response.json()
        if 'total_volumes' in vol_data and vol_data['total_volumes']:
            vol_df = pd.DataFrame(vol_data['total_volumes'], columns=["timestamp", "volume"])
            vol_df["date"] = pd.to_datetime(vol_df["timestamp"], unit='ms')
            vol_df = vol_df.set_index('date')
            
            # Resample volume to match OHLC data frequency
            vol_df = vol_df.resample('D')['volume'].sum().reset_index()
            
            # Merge volume data with OHLC data
            df = pd.merge_asof(df.sort_values('date'), 
                              vol_df.sort_values('date'), 
                              on='date', 
                              direction='nearest')
    
    # Handle missing volume data
    if 'volume' not in df.columns:
        df['volume'] = (df['high'] + df['low'] + df['close']) / 3  # Use typical price as volume proxy
    
    # Resample based on timeframe
    if timeframe != '1d':
        df = df.set_index('date')
        if timeframe in ['1m', '5m', '15m', '30m']:
            # For intraday, we need to simulate the data as CoinGecko doesn't provide minute data directly
            # This is a simplified approach - in production, consider using a different API
            df = df.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        df = df.reset_index()
    
    # Calculate additional metrics
    df['date_only'] = df['date'].dt.date
    df = df.sort_values('date')
    
    # Calculate price changes
    df['price_change'] = df['close'].pct_change() * 100
    df['price_change_abs'] = df['close'].diff()
    
    # Calculate typical price and money flow
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['money_flow'] = df['typical_price'] * df['volume']
    
    # Calculate technical indicators
    df = add_technical_indicators(df)
    
    return df

def add_technical_indicators(df):
    # Ensure we have enough data points
    if len(df) < 50:  # We need at least 50 points for reliable indicators
        return df
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    try:
        # Ensure we're working with numeric data only
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate indicators
        if len(numeric_df) >= 20:
            df['sma_20'] = SMAIndicator(close=df['close'], window=20, fillna=True).sma_indicator()
        
        if len(numeric_df) >= 50:
            df['ema_50'] = EMAIndicator(close=df['close'], window=50, fillna=True).ema_indicator()
        
        # RSI with minimum data check
        if len(numeric_df) >= 14:
            df['rsi'] = RSIIndicator(close=df['close'], window=14, fillna=True).rsi()
            df['rsi_peak'] = df['rsi'].rolling(5, center=True).max() == df['rsi']
            df['rsi_trough'] = df['rsi'].rolling(5, center=True).min() == df['rsi']
            df['rsi_signal'] = df['rsi'].rolling(window=9).mean()
        
        # MACD with minimum data check
        if len(numeric_df) >= 26:
            macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
            df['macd_above_signal'] = df['macd'] > df['macd_signal']
            df['macd_crossover'] = df['macd_above_signal'].ne(df['macd_above_signal'].shift())
        
        # Bollinger Bands
        if len(numeric_df) >= 20:
            bb = BollingerBands(close=df['close'], window=20, window_dev=2, fillna=True)
            df['bb_high'] = bb.bollinger_hband()
            df['bb_mid'] = bb.bollinger_mavg()
            df['bb_low'] = bb.bollinger_lband()
        
        # Volume Weighted Average Price
        if 'volume' in df.columns and 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Average True Range
        if len(numeric_df) >= 15:  # Need at least 14 periods + 1 for shift
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
        
        # Stochastic Oscillator
        if len(numeric_df) >= 14:
            low_min = df['low'].rolling(window=14).min()
            high_max = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-10))  # Add small number to avoid division by zero
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # On-Balance Volume
        if 'volume' in df.columns and 'close' in df.columns:
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Rate of Change
        if len(numeric_df) >= 10:  # Need at least 9 periods + 1 for pct_change
            df['roc'] = df['close'].pct_change(periods=9) * 100
        
        # Forward fill and backfill any remaining NaN values
        df = df.ffill().bfill()
        
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        # Return the original dataframe with any successfully calculated indicators
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

# Sidebar selectors
time_period = st.sidebar.select_slider(
    "Select Time Range",
    options=[7, 14, 30, 60, 90, 180, 365],
    value=30,
    format_func=lambda x: f"{x} days"
)
days = time_period  # Use the selected number of days

# Timeframe selector
timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    options=["1d", "4h", "1h", "15m"],
    index=0,
    help="Select the timeframe for the price data (1d = daily, 4h = 4 hours, 1h = 1 hour, 15m = 15 minutes)"
)

# Fetch market data with loading spinner
with st.spinner(f"Loading {selected_name} market data for {time_period} days ({timeframe} timeframe)..."):
    df = fetch_market_data(selected_id, days=days, timeframe=timeframe)

if df is not None and not df.empty:
    # Display Main Card with price and stats
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    try:
        # Latest price and basic stats
        latest_price = df.iloc[-1]["close"]
        first_price = df.iloc[0]["close"]
        price_change = latest_price - first_price
        pct_change = (price_change / first_price) * 100
        
        # Calculate 24h price change if we have enough data
        if len(df) >= 2:
            price_24h_ago = df.iloc[-2]["close"]
            price_change_24h = latest_price - price_24h_ago
            pct_change_24h = (price_change_24h / price_24h_ago) * 100
        
        # Get high and low for the selected period
        high_price = df["high"].max()
        low_price = df["low"].min()
        
        # Calculate 24h volume (or use latest available)
        volume_24h = df["volume"].iloc[-1] if len(df) > 0 else 0
        avg_volume = df["volume"].mean() if len(df) > 0 else 0
        
        # Prepare RSI data
        rsi_value = df["rsi"].iloc[-1] if "rsi" in df.columns and len(df) > 0 else None
        
        # Create metrics in a more organized way
        col1, col2, col3, col4 = st.columns(4)
        
        # Current Price Card
        with col1:
            st.metric(
                label="Current Price",
                value=f"${latest_price:,.2f}",
                delta=f"${price_change_24h:+.2f} ({pct_change_24h:+.2f}%)" if len(df) >= 2 else None,
                delta_color="normal"
            )
            st.caption(f"High: ${high_price:,.2f} | Low: ${low_price:,.2f}")
        
        # Price Change Card
        with col2:
            st.metric(
                label=f"{days}-Day Performance",
                value=f"{pct_change:+.2f}%",
                delta=f"${price_change:+,.2f}",
                delta_color="normal"
            )
            st.caption(f"From ${first_price:,.2f} to ${latest_price:,.2f}")
        
        # Volume Card
        with col3:
            volume_change = ((volume_24h - avg_volume) / avg_volume * 100) if avg_volume > 0 else 0
            st.metric(
                label="24h Trading Volume",
                value=f"${volume_24h:,.0f}",
                delta=f"{volume_change:+.1f}% vs avg" if avg_volume > 0 else None,
                delta_color="normal"
            )
            st.caption(f"Avg: ${avg_volume:,.0f}" if avg_volume > 0 else "No volume data")
        
        # RSI Card with visual indicator
        with col4:
            if rsi_value is not None:
                # Calculate RSI progress
                rsi_progress = min(max((rsi_value - 20) / 60, 0), 1)  # Normalize to 0-1 (20-80 range)
                
                # Determine RSI status and color
                if rsi_value < 30:
                    status = "Oversold"
                    color = "#4caf50"  # Green
                elif rsi_value > 70:
                    status = "Overbought"
                    color = "#f44336"  # Red
                else:
                    status = "Neutral"
                    color = "#ff9800"  # Orange
                
                # Create a visual indicator using HTML/CSS
                st.markdown(
                    f"""
                    <div style="margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span>RSI (14)</span>
                            <span style="font-weight: bold; color: {color};">{rsi_value:.2f}</span>
                        </div>
                        <div style="background: #2d2d2d; height: 8px; border-radius: 4px; overflow: hidden;">
                            <div style="width: {rsi_value}%; height: 100%; background: {color};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: #aaa; margin-top: 2px;">
                            <span>30</span>
                            <span>{status}</span>
                            <span>70</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.metric("RSI (14)", "N/A")
    except Exception as e:
        st.error(f"Error displaying price data: {str(e)}")
        st.stop()
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Technical Analysis", "📊 OHLC Details", "📋 Raw Data"])
    
    with tab1:
        # Overview tab with main chart and key metrics
        with st.container():
            st.markdown("### Price Chart")
            with st.spinner('Loading price chart...'):
                try:
                    st.plotly_chart(
                        plot_candlestick_chart(df), 
                        use_container_width=True,
                        theme="streamlit"  # Use Streamlit's theme for consistency
                    )
                except Exception as e:
                    st.warning("Could not display candlestick chart. Showing line chart instead.")
                    st.line_chart(
                        df.set_index('date')['close'],
                        use_container_width=True
                    )
        
        # Additional metrics in a grid
        st.markdown("### Market Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container():
                st.metric("24h High/Low", 
                         f"${df['high'].max():.2f} / ${df['low'].min():.2f}")
        
        with col2:
            with st.container():
                st.metric("24h Volume", 
                         f"${df['volume'].iloc[-1]:,.0f}",
                         f"{((df['volume'].iloc[-1] - df['volume'].mean()) / df['volume'].mean() * 100):+.1f}% vs avg" 
                         if len(df) > 1 else None)
        
        with col3:
            with st.container():
                st.metric("Market Sentiment", 
                         "Bullish" if df['close'].iloc[-1] > df['close'].iloc[0] else "Bearish",
                         f"{len(df[df['close'] > df['close'].shift(1)]) / len(df) * 100:.1f}% up days" 
                         if len(df) > 1 else None)
    
    with tab2:
        # Technical Analysis tab with multiple indicators
        st.markdown("### Technical Indicators")
        
        # Indicator selector
        indicators = ["All Indicators", "RSI", "MACD", "Volume"]
        selected_indicators = st.multiselect(
            "Select indicators to display:",
            options=indicators[1:],
            default=indicators[1:],
            key="indicator_selector"
        )
        
        # Show selected indicators
        with st.spinner('Loading technical indicators...'):
            try:
                if not selected_indicators or "All Indicators" in selected_indicators or "RSI" in selected_indicators:
                    if 'rsi' in df.columns and len(df) > 0:
                        st.plotly_chart(
                            plot_rsi_chart(df), 
                            use_container_width=True
                        )
                
                if not selected_indicators or "All Indicators" in selected_indicators or "MACD" in selected_indicators:
                    if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_hist']):
                        st.plotly_chart(
                            plot_macd_chart(df), 
                            use_container_width=True
                        )
                
                if not selected_indicators or "All Indicators" in selected_indicators or "Volume" in selected_indicators:
                    if 'volume' in df.columns and len(df) > 0:
                        st.plotly_chart(
                            plot_volume_chart(df), 
                            use_container_width=True
                        )
                
                if not selected_indicators:
                    st.info("Select indicators to display from the dropdown above.")
                
            except Exception as e:
                st.error(f"Error displaying technical indicators: {str(e)}")
                st.warning("Some technical indicators could not be displayed. Please try a different time period or cryptocurrency.")
    
    with tab3:
        # OHLC Details tab
        st.markdown("### OHLC Price Details")
        
        # Show OHLC metrics for the latest data point
        if not df.empty:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # Create metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                delta_open = (latest['open'] - prev['open']) / prev['open'] * 100
                st.metric("Open", f"${latest['open']:,.2f}", 
                         f"{delta_open:+.2f}%" if len(df) > 1 else None)
            
            with col2:
                st.metric("High", f"${latest['high']:,.2f}")
            
            with col3:
                st.metric("Low", f"${latest['low']:,.2f}")
            
            with col4:
                delta_close = (latest['close'] - prev['close']) / prev['close'] * 100
                st.metric("Close", f"${latest['close']:,.2f}", 
                         f"{delta_close:+.2f}%" if len(df) > 1 else None)
            
            # Show OHLC chart
            fig_ohlc = go.Figure()
            
            # Candlestick
            fig_ohlc.add_trace(go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ))
            
            # Update layout
            fig_ohlc.update_layout(
                title=f"{selected_name} OHLC Chart ({timeframe} timeframe)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template='plotly_dark',
                showlegend=False,
                height=500,
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig_ohlc, use_container_width=True)
            
            # Show volume chart
            st.markdown("### Volume Analysis")
            
            # Calculate volume metrics
            avg_volume = df['volume'].mean()
            latest_volume = latest['volume']
            volume_change = (latest_volume - avg_volume) / avg_volume * 100
            
            # Create volume metrics
            vol_col1, vol_col2, vol_col3 = st.columns(3)
            
            with vol_col1:
                st.metric("24h Volume", f"${latest_volume:,.0f}", 
                         f"{volume_change:+.1f}% vs avg")
            
            with vol_col2:
                st.metric("Volume Avg", f"${avg_volume:,.0f}")
            
            with vol_col3:
                vol_trend = "📈" if latest_volume > avg_volume else "📉"
                st.metric("Volume Trend", vol_trend)
            
            # Volume chart with price-based coloring
            fig_vol = go.Figure()
            
            # Add volume bars with color based on price movement
            colors = ['#26a69a' if close >= open_ else '#ef5350' 
                     for close, open_ in zip(df['close'], df['open'])]
            
            fig_vol.add_trace(go.Bar(
                x=df['date'],
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7,
                hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
            ))
            
            # Add average volume line
            fig_vol.add_hline(y=avg_volume, line_dash='dash', 
                            line_color='#9e9e9e',
                            annotation_text=f'Avg: ${avg_volume:,.0f}',
                            annotation_position='top right')
            
            # Update layout
            fig_vol.update_layout(
                title="Trading Volume with Price-based Coloring",
                xaxis_title="Date",
                yaxis_title="Volume",
                template='plotly_dark',
                showlegend=False,
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
    
    with tab4:
        # Raw Data tab with filtering options
        st.markdown("### Market Data")
        
        # Ensure date column is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Get min and max dates
        min_date = df['date'].min().to_pydatetime()
        max_date = df['date'].max().to_pydatetime()
        
        # Convert dates to timestamps for the slider
        min_ts = int(min_date.timestamp())
        max_ts = int(max_date.timestamp())
        
        # Create date slider
        selected_ts = st.slider(
            "Select date range:",
            min_value=min_ts,
            max_value=max_ts,
            value=(min_ts, max_ts),
            format="YYYY/MM/DD",
            key="date_slider_raw"  # Unique key for this slider
        )
        
        # Convert timestamps back to datetime
        start_date = pd.to_datetime(selected_ts[0], unit='s')
        end_date = pd.to_datetime(selected_ts[1], unit='s')
        
        # Filter data
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # Show data table with pagination
        st.dataframe(
            filtered_df.drop(columns=['date_only']).set_index('date').sort_index(ascending=False),
            use_container_width=True,
            height=500,
            column_config={
                'open': st.column_config.NumberColumn(format='$%.2f'),
                'high': st.column_config.NumberColumn(format='$%.2f'),
                'low': st.column_config.NumberColumn(format='$%.2f'),
                'close': st.column_config.NumberColumn(format='$%.2f'),
                'volume': st.column_config.NumberColumn(format='%.2f')
            }
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"crypto_data_{selected_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
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

