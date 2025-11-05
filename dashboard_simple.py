"""
Ultra RL Trading Dashboard - Simplified Version
Works around PyTorch compatibility issues
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time

# Page configuration
st.set_page_config(
    page_title="Ultra RL Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .stMetric {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #1c83e1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio_value' not in st.session_state:
    st.session_state.portfolio_value = 100000
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'equity_history' not in st.session_state:
    st.session_state.equity_history = [100000]
if 'current_position' not in st.session_state:
    st.session_state.current_position = 0

def generate_market_data(n_days=100):
    """Generate simulated market data"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generate realistic OHLCV data
    close = 100
    data = []
    
    for date in dates:
        change = np.random.normal(0, 2)
        close = close * (1 + change/100)
        
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = close * (1 + np.random.normal(0, 0.005))
        
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def calculate_indicators(df):
    """Calculate technical indicators"""
    # Simple Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    return df

def simulate_trading_step():
    """Simulate one trading step"""
    # Random action (for demo)
    action = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4])
    
    # Update portfolio
    price_change = np.random.normal(0, 0.02)
    st.session_state.portfolio_value *= (1 + price_change)
    st.session_state.equity_history.append(st.session_state.portfolio_value)
    
    # Record trade
    if action != 'HOLD':
        st.session_state.trades.append({
            'time': datetime.now(),
            'action': action,
            'price': 100 * (1 + price_change),
            'value': st.session_state.portfolio_value
        })
    
    return action, price_change

# Header
st.title("🚀 Ultra RL Trading Dashboard")
st.markdown("### Live AI Trading Performance Monitor")

# Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    if st.button("🔄 Refresh Data", type="primary"):
        st.rerun()
    
    st.divider()
    
    # Trading controls
    st.subheader("📈 Trading Settings")
    
    risk_level = st.select_slider(
        "Risk Level",
        options=["Conservative", "Moderate", "Aggressive"],
        value="Moderate"
    )
    
    position_size = st.slider(
        "Max Position Size (%)",
        min_value=10,
        max_value=100,
        value=50,
        step=10
    )
    
    st.divider()
    
    # Quick stats
    st.subheader("📊 Performance")
    returns = (st.session_state.portfolio_value - 100000) / 100000 * 100
    st.metric("Total Return", f"{returns:.2f}%")
    st.metric("Portfolio Value", f"${st.session_state.portfolio_value:,.0f}")
    st.metric("Total Trades", len(st.session_state.trades))

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📈 Market", "💰 Portfolio", "🤖 AI Performance", "📊 Analytics"])

with tab1:
    st.subheader("Market Overview")
    
    # Generate and display market data
    df = generate_market_data(200)
    df = calculate_indicators(df)
    
    # Candlestick chart
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price Action', 'RSI', 'Volume')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Add SMA lines
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['SMA_20'], name='SMA 20', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['SMA_50'], name='SMA 50', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['BB_upper'], name='BB Upper', 
                  line=dict(color='gray', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['BB_lower'], name='BB Lower',
                  line=dict(color='gray', dash='dash')),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['RSI'], name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    fig.add_trace(
        go.Bar(x=df['date'], y=df['volume'], name='Volume'),
        row=3, col=1
    )
    
    fig.update_layout(height=700, showlegend=True, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Portfolio Performance")
    
    col1, col2, col3 = st.columns(3)
    
    # Simulate a trading step for demo
    action, change = simulate_trading_step()
    
    with col1:
        st.metric("Last Action", action)
    with col2:
        st.metric("Last Change", f"{change*100:.2f}%")
    with col3:
        st.metric("Position", "LONG" if st.session_state.current_position > 0 else "FLAT")
    
    # Equity curve
    equity_df = pd.DataFrame({
        'Step': range(len(st.session_state.equity_history)),
        'Equity': st.session_state.equity_history
    })
    
    fig = px.line(equity_df, x='Step', y='Equity', title='Portfolio Equity Curve')
    fig.add_hline(y=100000, line_dash="dash", line_color="white")
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent trades
    if st.session_state.trades:
        st.subheader("Recent Trades")
        recent_trades = pd.DataFrame(st.session_state.trades[-10:])
        st.dataframe(recent_trades)

with tab3:
    st.subheader("AI Agent Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Action distribution
        actions_data = pd.DataFrame({
            'Action': ['BUY', 'SELL', 'HOLD'],
            'Count': [30, 25, 45]
        })
        
        fig = px.pie(actions_data, values='Count', names='Action', 
                    title='Action Distribution')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Win rate gauge
        win_rate = 55  # Demo value
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=win_rate,
            title={'text': "Win Rate (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green" if win_rate > 50 else "red"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Risk Analytics")
    
    # Calculate metrics
    returns_array = np.diff(st.session_state.equity_history) / st.session_state.equity_history[:-1]
    sharpe = np.mean(returns_array) / (np.std(returns_array) + 1e-6) * np.sqrt(252)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with col2:
        st.metric("Max Drawdown", "-5.3%")  # Demo value
    with col3:
        st.metric("Volatility", "12.5%")  # Demo value
    with col4:
        st.metric("Beta", "0.85")  # Demo value
    
    # Returns distribution
    returns_data = np.random.normal(0.001, 0.02, 100)
    fig = px.histogram(returns_data * 100, nbins=30, 
                      title='Returns Distribution (%)',
                      labels={'value': 'Return (%)', 'count': 'Frequency'})
    fig.update_layout(template='plotly_dark', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.markdown(
    """<div style='text-align: center'>
    <p>Ultra RL Trading System | Real-time AI Trading Dashboard</p>
    </div>""",
    unsafe_allow_html=True
)
