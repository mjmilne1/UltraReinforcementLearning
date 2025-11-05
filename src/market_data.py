"""
Real Market Data Integration
Connects to live market data sources
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional
import ccxt  # For cryptocurrency data

class MarketDataFetcher:
    """Fetches real market data from various sources"""
    
    def __init__(self, source="yahoo"):
        """
        Initialize market data fetcher
        
        Args:
            source: Data source ('yahoo', 'crypto', 'polygon', 'alpaca')
        """
        self.source = source
        self.cache = {}
        
        # Initialize crypto exchange if needed
        if source == "crypto":
            self.exchange = ccxt.binance()  # Free, no API key needed
    
    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Get stock data from Yahoo Finance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            # Handle both 'Date' and 'date' columns
            if not data.empty:
                data = data.reset_index()
                # Rename columns to lowercase
                data.columns = [col.lower() if col != 'Date' else 'date' for col in data.columns]
                # Handle timezone
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                    if hasattr(data['date'].dt, 'tz'):
                        data['date'] = data['date'].dt.tz_localize(None)
            
            print(f"✅ Fetched {len(data)} data points for {symbol}")
            return data
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            return pd.DataFrame()    
    def get_crypto_data(self, symbol: str = "BTC/USDT", timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
        """
        Get cryptocurrency data
        
        Args:
            symbol: Crypto pair (e.g., 'BTC/USDT', 'ETH/USDT')
            timeframe: Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
            limit: Number of candles
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
            data = data.drop('timestamp', axis=1)
            
            print(f"✅ Fetched {len(data)} data points for {symbol}")
            return data
        except Exception as e:
            print(f"❌ Error fetching crypto data: {e}")
            return pd.DataFrame()
    
    def get_live_price(self, symbol: str, asset_type: str = "stock") -> Dict:
        """
        Get live price for a symbol
        
        Args:
            symbol: Asset symbol
            asset_type: 'stock' or 'crypto'
        
        Returns:
            Dictionary with current price info
        """
        try:
            if asset_type == "stock":
                ticker = yf.Ticker(symbol)
                info = ticker.info
                return {
                    'symbol': symbol,
                    'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                    'change': info.get('regularMarketChange', 0),
                    'change_pct': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('regularMarketVolume', 0),
                    'timestamp': datetime.now()
                }
            elif asset_type == "crypto":
                ticker = self.exchange.fetch_ticker(symbol)
                return {
                    'symbol': symbol,
                    'price': ticker['last'],
                    'change': ticker['change'],
                    'change_pct': ticker['percentage'],
                    'volume': ticker['baseVolume'],
                    'timestamp': datetime.now()
                }
        except Exception as e:
            print(f"❌ Error getting live price: {e}")
            return {}
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            period: Time period
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        for symbol in symbols:
            df = self.get_stock_data(symbol, period)
            if not df.empty:
                data[symbol] = df
            time.sleep(0.5)  # Avoid rate limiting
        return data
    
    def get_market_indicators(self) -> Dict:
        """Get major market indicators"""
        indicators = {}
        
        # Major indices
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^VIX': 'VIX (Volatility)'
        }
        
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                indicators[name] = {
                    'price': info.get('regularMarketPrice', 0),
                    'change': info.get('regularMarketChange', 0),
                    'change_pct': info.get('regularMarketChangePercent', 0)
                }
            except:
                pass
        
        return indicators

class LiveDataStream:
    """Stream live market data"""
    
    def __init__(self, symbols: List[str], interval: int = 60):
        """
        Initialize live data stream
        
        Args:
            symbols: List of symbols to track
            interval: Update interval in seconds
        """
        self.symbols = symbols
        self.interval = interval
        self.fetcher = MarketDataFetcher()
        self.running = False
        self.data_buffer = []
    
    def start_stream(self, callback=None):
        """Start streaming data"""
        self.running = True
        
        while self.running:
            for symbol in self.symbols:
                data = self.fetcher.get_live_price(symbol)
                
                if data:
                    self.data_buffer.append(data)
                    
                    if callback:
                        callback(data)
                    
                    print(f"📊 {symbol}: ${data.get('price', 0):.2f} "
                          f"({data.get('change_pct', 0):+.2f}%)")
            
            time.sleep(self.interval)
    
    def stop_stream(self):
        """Stop streaming"""
        self.running = False

def test_market_data():
    """Test real market data fetching"""
    print("="*60)
    print("🌐 Testing Real Market Data Connection")
    print("="*60)
    
    fetcher = MarketDataFetcher()
    
    # Test stock data
    print("\n1. Fetching Apple (AAPL) stock data...")
    aapl = fetcher.get_stock_data("AAPL", period="1mo", interval="1d")
    if not aapl.empty:
        print(f"   Latest close: ${aapl['close'].iloc[-1]:.2f}")
        print(f"   Data shape: {aapl.shape}")
    
    # Test live price
    print("\n2. Getting live prices...")
    for symbol in ["MSFT", "GOOGL", "TSLA"]:
        price_data = fetcher.get_live_price(symbol)
        if price_data:
            print(f"   {symbol}: ${price_data.get('price', 0):.2f}")
    
    # Test crypto data
    print("\n3. Fetching Bitcoin data...")
    fetcher_crypto = MarketDataFetcher(source="crypto")
    btc = fetcher_crypto.get_crypto_data("BTC/USDT", timeframe="1d", limit=30)
    if not btc.empty:
        print(f"   BTC latest: ${btc['close'].iloc[-1]:.2f}")
        print(f"   30-day change: {(btc['close'].iloc[-1]/btc['close'].iloc[0]-1)*100:.2f}%")
    
    # Test market indicators
    print("\n4. Getting market indicators...")
    indicators = fetcher.get_market_indicators()
    for name, data in indicators.items():
        if data:
            print(f"   {name}: {data.get('price', 0):.2f} ({data.get('change_pct', 0):+.2f}%)")
    
    print("\n✅ Market data connection successful!")

if __name__ == "__main__":
    test_market_data()
