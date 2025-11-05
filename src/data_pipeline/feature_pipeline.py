"""Feature Pipeline - Processes data from Kafka"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
import asyncio

from ..schemas.market_data import MarketData, Quote, Trade, OHLCV
import structlog

logger = structlog.get_logger()

class FeaturePipeline:
    """Converts raw market data into features for RL agents"""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize feature pipeline
        
        Args:
            window_size: Rolling window for feature calculation
        """
        self.window_size = window_size
        self.data_buffer = {}  # Symbol -> deque of data points
        self.features = {}  # Symbol -> feature vector
        
    def process(self, data: MarketData) -> Optional[np.ndarray]:
        """
        Process market data and generate features
        
        Args:
            data: Market data object
            
        Returns:
            Feature vector if enough data, None otherwise
        """
        symbol = data.symbol
        
        # Initialize buffer for new symbols
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = deque(maxlen=self.window_size)
        
        # Add data to buffer
        self.data_buffer[symbol].append(data)
        
        # Generate features if we have enough data
        if len(self.data_buffer[symbol]) >= self.window_size:
            features = self._generate_features(symbol)
            self.features[symbol] = features
            return features
        
        return None
    
    def _generate_features(self, symbol: str) -> np.ndarray:
        """Generate feature vector for a symbol"""
        data = list(self.data_buffer[symbol])
        
        features = []
        
        # Price features
        prices = [d.last if hasattr(d, 'last') else d.close 
                 for d in data if hasattr(d, 'last') or hasattr(d, 'close')]
        
        if prices:
            # Basic statistics
            features.extend([
                np.mean(prices),
                np.std(prices),
                np.min(prices),# Create feature pipeline that processes consumed data
@'
"""Feature Pipeline - Processes data from Kafka"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
import asyncio

from ..schemas.market_data import MarketData, Quote, Trade, OHLCV
import structlog

logger = structlog.get_logger()

class FeaturePipeline:
    """Converts raw market data into features for RL agents"""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize feature pipeline
        
        Args:
            window_size: Rolling window for feature calculation
        """
        self.window_size = window_size
        self.data_buffer = {}  # Symbol -> deque of data points
        self.features = {}  # Symbol -> feature vector
        
    def process(self, data: MarketData) -> Optional[np.ndarray]:
        """
        Process market data and generate features
        
        Args:
            data: Market data object
            
        Returns:
            Feature vector if enough data, None otherwise
        """
        symbol = data.symbol
        
        # Initialize buffer for new symbols
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = deque(maxlen=self.window_size)
        
        # Add data to buffer
        self.data_buffer[symbol].append(data)
        
        # Generate features if we have enough data
        if len(self.data_buffer[symbol]) >= self.window_size:
            features = self._generate_features(symbol)
            self.features[symbol] = features
            return features
        
        return None
    
    def _generate_features(self, symbol: str) -> np.ndarray:
        """Generate feature vector for a symbol"""
        data = list(self.data_buffer[symbol])
        
        features = []
        
        # Price features
        prices = [d.last if hasattr(d, 'last') else d.close 
                 for d in data if hasattr(d, 'last') or hasattr(d, 'close')]
        
        if prices:
            # Basic statistics
            features.extend([
                np.mean(prices),
                np.std(prices),
                np.min(prices),
                np.max(prices),
                prices[-1] / prices[0] - 1,  # Return
            ])
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                if len(prices) >= period:
                    ma = np.mean(prices[-period:])
                    features.append(prices[-1] / ma - 1)
                else:
                    features.append(0)
            
            # RSI
            rsi = self._calculate_rsi(prices)
            features.append(rsi)
            
            # Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(prices)
            features.extend([
                (prices[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5,
                (bb_upper - bb_lower) / prices[-1] if prices[-1] != 0 else 0
            ])
        
        # Volume features
        volumes = [d.volume for d in data if hasattr(d, 'volume') and d.volume]
        if volumes:
            features.extend([
                np.mean(volumes),
                np.std(volumes),
                volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 1
            ])
        
        # Spread features (if quotes available)
        quotes = [d for d in data if isinstance(d, Quote)]
        if quotes:
            spreads = [(q.ask - q.bid) / q.bid for q in quotes if q.bid > 0]
            if spreads:
                features.extend([
                    np.mean(spreads),
                    np.std(spreads),
                    spreads[-1]
                ])
        
        # Pad or truncate to fixed size
        feature_vector = np.array(features)
        
        # Ensure fixed dimensionality (pad with zeros if needed)
        target_dim = 256  # For DQN agent
        if len(feature_vector) < target_dim:
            feature_vector = np.pad(feature_vector, (0, target_dim - len(feature_vector)))
        elif len(feature_vector) > target_dim:
            feature_vector = feature_vector[:target_dim]
        
        return feature_vector
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(
        self, 
        prices: List[float], 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> tuple:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return prices[-1], prices[-1]
        
        recent_prices = prices[-period:]
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, lower_band
    
    def get_state_vector(self, symbols: List[str]) -> np.ndarray:
        """
        Get combined state vector for multiple symbols
        
        Args:
            symbols: List of symbols
            
        Returns:
            Combined state vector
        """
        states = []
        
        for symbol in symbols:
            if symbol in self.features:
                states.append(self.features[symbol])
            else:
                # Use zero vector if no features yet
                states.append(np.zeros(256))
        
        # Stack and flatten
        if states:
            return np.concatenate(states)
        else:
            return np.zeros(256 * len(symbols))

class RealTimeFeatureEngine:
    """Real-time feature engine that connects Kafka to RL agents"""
    
    def __init__(self):
        self.pipeline = FeaturePipeline()
        self.consumer = None
        self.feature_buffer = asyncio.Queue(maxsize=1000)
        
    async def start(self):
        """Start the feature engine"""
        from ..consumers.market_data_consumer import MarketDataConsumer
        
        # Create consumer with custom handler
        self.consumer = MarketDataConsumer(
            message_handler=self.process_market_data
        )
        
        # Start consumer in background
        consumer_task = asyncio.create_task(
            self.consumer.start_async()
        )
        
        # Start feature processor
        processor_task = asyncio.create_task(
            self.process_features()
        )
        
        await asyncio.gather(consumer_task, processor_task)
    
    def process_market_data(self, data: MarketData, key: str, topic: str):
        """Process incoming market data"""
        # Generate features
        features = self.pipeline.process(data)
        
        if features is not None:
            # Add to feature buffer
            try:
                self.feature_buffer.put_nowait({
                    'symbol': data.symbol,
                    'features': features,
                    'timestamp': data.timestamp
                })
            except asyncio.QueueFull:
                logger.warning("Feature buffer full, dropping data")
    
    async def process_features(self):
        """Process features and feed to RL agents"""
        while True:
            try:
                # Get feature from buffer
                feature_data = await self.feature_buffer.get()
                
                # Here you would feed to your RL agent
                logger.info(
                    "Generated features",
                    symbol=feature_data['symbol'],
                    feature_dim=len(feature_data['features'])
                )
                
                # TODO: Send to DQN/PPO/A3C agents
                
            except Exception as e:
                logger.error(f"Feature processing error: {e}")
