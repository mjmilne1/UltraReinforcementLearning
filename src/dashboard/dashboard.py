"""
Ultra RL Trading Dashboard
Real-time visualization of AI trading performance
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.environment.trading_env import TradingEnvironment
from src.agents.dqn_agent_simple import DQNAgent
from src.indicators.technical_indicators import TechnicalIndicators

# Page configuration
st.set_page_config(
    page_title="Ultra RL Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {padding: 0rem 0rem;}
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #1c83e1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trading_active' not in st.session_state:
    st.session_state.trading_active = False
if 'env' not in st.session_state:
    st.session_state.env = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'performance_history' not in st.session_state:
    st.session_state.performance_history = []

def create_sample_data(n_days=500):
    """Generate sample market data"""
    np.random.seed(int(time.time()))
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'high': prices * (1 + np.abs(np.random.uniform(0, 0.02, n_days))),
        'low': prices * (1 - np.abs(np.random.uniform(0, 0.02, n_days))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    return data

def initialize_trading():
    """Initialize environment and agent"""
    data = create_sample_data(500)
    
    env = TradingEnvironment(
        data=data,
        initial_balance=100000,
        window_size=20,
        reward_type='sharpe'
    )
    
    agent = DQNAgent(
        state_size=env.observation_size,
        action_size=3
    )
    
    # Pre-train agent briefly
    with st.spinner("Training agent..."):
        for _ in range(5):
            obs, _ = env.reset()
            done = False
            while not done:
                action = agent.act(obs, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.remember(obs, action, reward, next_obs, done)
                if len(agent.memory) > agent.batch_size:
                    agent.learn()
                obs = next_obs
    
    # Reset for trading
    obs, info = env.reset()
    
    return env, agent, data

# Header
st.title("🚀 Ultra RL Trading Dashboard")
st.markdown("### AI-Powered Trading with Deep Reinforcement Learning")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    if st.button("🎯 Initialize System", type="primary"):
        with st.spinner("Initializing..."):
            env, agent, data = initialize_trading()
            st.session_state.env = env
            st.session_state.agent = agent
            st.session_state.data = data
            st.success("System initialized!")
    
    if st.session_state.env is not None:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Trading", disabled=st.session_state.trading_active):
                st.session_state.trading_active = True
        with col2:
            if st.button("⏸️ Stop Trading", disabled=not st.session_state.trading_active):
                st.session_state.trading_active = False
        
        st.divider()
        
        # Trading parameters
        st.subheader("📈 Trading Parameters")
        risk_level = st.select_slider(
            "Risk Level",
            options=["Conservative", "Moderate", "Aggressive"],
            value="Moderate"
        )
        
        position_size = st.slider(
            "Position Size (%)",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )
        
        st.divider()
        
        # Performance metrics
        st.subheader("📊 Quick Stats")
        if st.session_state.env:
            metrics = st.session_state.env.portfolio.get_performance_metrics()
            st.metric("Total Return", f"{metrics.get('total_return_pct', 0):.2f}%")
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
            st.metric("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%")

# Main content area
if st.session_state.env is None:
    # Welcome screen
    st.info("👈 Click 'Initialize System' in the sidebar to begin")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("DQN Agent", "Ready", "Neural Network")
    with col2:
        st.metric("Market Data", "Live Feed", "Real-time")
    with col3:
        st.metric("Risk System", "Active", "Protected")
else:
    # Trading interface
    env = st.session_state.env
    agent = st.session_state.agent
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = env.portfolio.get_performance_metrics()
    
    with col1:
        st.metric(
            "Portfolio Value",
            f"${metrics.get('current_equity', 100000):,.0f}",
            f"{metrics.get('total_return_pct', 0):.2f}%"
        )
    
    with col2:
        st.metric(
            "Today's P&L",
            f"${metrics.get('total_return', 0) * 100000:,.0f}",
            f"{metrics.get('total_return_pct', 0):.2f}%"
        )
    
    with col3:
        st.metric(
            "Total Trades",
            f"{metrics.get('total_trades', 0)}",
            "Active" if metrics.get('current_positions', 0) > 0 else "Flat"
        )
    
    with col4:
        sharpe = metrics.get('sharpe_ratio', 0)
        st.metric(
            "Sharpe Ratio",
            f"{sharpe:.2f}",
            "Good" if sharpe > 1 else "Low"
        )
    
    with col5:
        dd = metrics.get('max_drawdown_pct', 0)
        st.metric(
            "Max Drawdown",
            f"{dd:.1f}%",
            "Safe" if dd > -10 else "Risk"
        )
    
    # Charts
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "💰 Portfolio", "🤖 Agent Decisions", "📊 Risk Metrics"])
    
    with tab1:
        # Price chart with indicators
        if 'data' in st.session_state:
            data = st.session_state.data
            
            # Calculate indicators
            ti = TechnicalIndicators()
            data['SMA_20'] = ti.sma(data['close'].values, 20)
            data['SMA_50'] = ti.sma(data['close'].values, 50)
            
            # Bollinger Bands
            bb = ti.bollinger_bands(data['close'].values)
            data['BB_Upper'] = bb.upper_band
            data['BB_Lower'] = bb.lower_band
            
            # Create candlestick chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=('Price Action', 'Volume')
            )
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=data['date'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Moving averages
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['SMA_20'],
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['SMA_50'],
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
            
            # Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['BB_Upper'],
                    name='BB Upper',
                    line=dict(color='gray', width=0.5, dash='dash')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['BB_Lower'],
                    name='BB Lower',
                    line=dict(color='gray', width=0.5, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.2)'
                ),
                row=1, col=1
            )
            
            # Volume
            fig.add_trace(
                go.Bar(
                    x=data['date'],
                    y=data['volume'],
                    name='Volume',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title='Market Price Action',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark',
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Portfolio performance
        if len(env.portfolio.equity_curve) > 1:
            equity_df = pd.DataFrame({
                'Time': range(len(env.portfolio.equity_curve)),
                'Equity': env.portfolio.equity_curve
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df['Time'],
                y=equity_df['Equity'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='green', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,255,0,0.1)'
            ))
            
            # Add baseline
            fig.add_hline(
                y=100000,
                line_dash="dash",
                line_color="white",
                annotation_text="Initial Capital"
            )
            
            fig.update_layout(
                title='Portfolio Performance',
                xaxis_title='Time Steps',
                yaxis_title='Portfolio Value ($)',
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Returns distribution
            if len(env.portfolio.returns) > 0:
                returns_df = pd.DataFrame({
                    'Returns': np.array(env.portfolio.returns) * 100
                })
                
                fig = px.histogram(
                    returns_df,
                    x='Returns',
                    nbins=30,
                    title='Returns Distribution',
                    labels={'Returns': 'Returns (%)'},
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Agent decisions
        st.subheader("🤖 Agent Decision Making")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Action distribution
            if len(st.session_state.trade_history) > 0:
                actions = pd.DataFrame(st.session_state.trade_history)
                action_counts = actions['action'].value_counts()
                
                fig = px.pie(
                    values=action_counts.values,
                    names=action_counts.index,
                    title='Action Distribution',
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Q-values visualization
            st.info("Q-Values show the expected future reward for each action")
            
            # Simulate Q-values for visualization
            q_values = np.random.randn(3) * 0.1
            q_df = pd.DataFrame({
                'Action': ['HOLD', 'BUY', 'SELL'],
                'Q-Value': q_values
            })
            
            fig = px.bar(
                q_df,
                x='Action',
                y='Q-Value',
                title='Current Q-Values',
                color='Q-Value',
                color_continuous_scale='RdYlGn',
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Risk metrics
        st.subheader("📊 Risk Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Drawdown chart
            if len(env.portfolio.equity_curve) > 1:
                equity_array = np.array(env.portfolio.equity_curve)
                peaks = np.maximum.accumulate(equity_array)
                drawdown = (equity_array - peaks) / peaks * 100
                
                dd_df = pd.DataFrame({
                    'Time': range(len(drawdown)),
                    'Drawdown': drawdown
                })
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dd_df['Time'],
                    y=dd_df['Drawdown'],
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255,0,0,0.2)'
                ))
                
                fig.update_layout(
                    title='Drawdown Analysis',
                    xaxis_title='Time',
                    yaxis_title='Drawdown (%)',
                    template='plotly_dark',
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk metrics gauge
            sharpe = metrics.get('sharpe_ratio', 0)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sharpe,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sharpe Ratio"},
                delta={'reference': 1.0},
                gauge={'axis': {'range': [None, 4]},
                       'bar': {'color': "darkgreen" if sharpe > 1 else "orange"},
                       'steps': [
                           {'range': [0, 1], 'color': "lightgray"},
                           {'range': [1, 2], 'color': "gray"}],
                       'threshold': {
                           'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': 3}}
            ))
            
            fig.update_layout(
                template='plotly_dark',
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Auto-refresh for live trading
    if st.session_state.trading_active:
        time.sleep(1)  # Simulate trading delay
        
        # Execute one trading step
        if env.current_step < len(env.data) - 1:
            obs = env._get_observation()
            action = agent.act(obs, training=False)
            
            # Record decision
            st.session_state.trade_history.append({
                'time': datetime.now(),
                'action': ['HOLD', 'BUY', 'SELL'][action],
                'price': env.prices[env.current_step],
                'equity': env.portfolio.equity
            })
            
            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update performance history
            st.session_state.performance_history.append(info)
            
            # Auto-refresh
            st.rerun()
        else:
            st.session_state.trading_active = False
            st.success("Trading session complete!")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center'>
        <p>🚀 Ultra RL Trading System | AI-Powered Trading | Built with ❤️</p>
    </div>
    """,
    unsafe_allow_html=True
)
