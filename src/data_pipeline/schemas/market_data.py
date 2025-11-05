"""Market Data Schemas for Kafka Messages"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

class AssetClass(str, Enum):
    """Asset class enumeration"""
    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"
    DERIVATIVE = "derivative"

class MarketDataType(str, Enum):
    """Type of market data"""
    QUOTE = "quote"
    TRADE = "trade"
    ORDER_BOOK = "order_book"
    OHLCV = "ohlcv"

class MarketData(BaseModel):
    """Base market data model"""
    
    # Identifiers
    symbol: str = Field(..., description="Trading symbol")
    exchange: str = Field(..., description="Exchange code")
    asset_class: AssetClass
    
    # Timestamps
    timestamp: datetime = Field(..., description="Data timestamp")
    received_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Data type
    data_type: MarketDataType
    
    # Sequence for ordering
    sequence: Optional[int] = Field(None, description="Sequence number")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Quote(MarketData):
    """Quote/Level 1 market data"""
    
    data_type: MarketDataType = MarketDataType.QUOTE
    
    # Price data
    bid: float = Field(..., gt=0)
    ask: float = Field(..., gt=0)
    bid_size: int = Field(..., ge=0)
    ask_size: int = Field(..., ge=0)
    
    # Additional info
    last: Optional[float] = Field(None, gt=0)
    volume: Optional[int] = Field(None, ge=0)
    
    @validator('ask')
    def validate_spread(cls, ask, values):
        """Ensure positive spread"""
        if 'bid' in values and ask <= values['bid']:
            raise ValueError(f"Ask {ask} must be greater than bid {values['bid']}")
        return ask

class Trade(MarketData):
    """Trade/Transaction data"""
    
    data_type: MarketDataType = MarketDataType.TRADE
    
    # Trade details
    price: float = Field(..., gt=0)
    quantity: int = Field(..., gt=0)
    trade_id: str = Field(...)
    
    # Flags
    is_buy: bool = Field(...)
    is_block: bool = Field(default=False)

class OHLCV(MarketData):
    """OHLCV candlestick data"""
    
    data_type: MarketDataType = MarketDataType.OHLCV
    
    # OHLCV data
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: int = Field(..., ge=0)
    
    # Time period
    interval: str = Field(..., regex="^[1-9][0-9]*[smhd]$")
    period_start: datetime
    period_end: datetime
    
    @validator('high')
    def validate_high(cls, high, values):
        """Ensure high is highest price"""
        if 'low' in values and hig
# Create market data models
@'
"""Market Data Schemas for Kafka Messages"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

class AssetClass(str, Enum):
    """Asset class enumeration"""
    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"
    DERIVATIVE = "derivative"

class MarketDataType(str, Enum):
    """Type of market data"""
    QUOTE = "quote"
    TRADE = "trade"
    ORDER_BOOK = "order_book"
    OHLCV = "ohlcv"

class MarketData(BaseModel):
    """Base market data model"""
    
    # Identifiers
    symbol: str = Field(..., description="Trading symbol")
    exchange: str = Field(..., description="Exchange code")
    asset_class: AssetClass
    
    # Timestamps
    timestamp: datetime = Field(..., description="Data timestamp")
    received_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Data type
    data_type: MarketDataType
    
    # Sequence for ordering
    sequence: Optional[int] = Field(None, description="Sequence number")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Quote(MarketData):
    """Quote/Level 1 market data"""
    
    data_type: MarketDataType = MarketDataType.QUOTE
    
    # Price data
    bid: float = Field(..., gt=0)
    ask: float = Field(..., gt=0)
    bid_size: int = Field(..., ge=0)
    ask_size: int = Field(..., ge=0)
    
    # Additional info
    last: Optional[float] = Field(None, gt=0)
    volume: Optional[int] = Field(None, ge=0)
    
    @validator('ask')
    def validate_spread(cls, ask, values):
        """Ensure positive spread"""
        if 'bid' in values and ask <= values['bid']:
            raise ValueError(f"Ask {ask} must be greater than bid {values['bid']}")
        return ask

class Trade(MarketData):
    """Trade/Transaction data"""
    
    data_type: MarketDataType = MarketDataType.TRADE
    
    # Trade details
    price: float = Field(..., gt=0)
    quantity: int = Field(..., gt=0)
    trade_id: str = Field(...)
    
    # Flags
    is_buy: bool = Field(...)
    is_block: bool = Field(default=False)

class OHLCV(MarketData):
    """OHLCV candlestick data"""
    
    data_type: MarketDataType = MarketDataType.OHLCV
    
    # OHLCV data
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: int = Field(..., ge=0)
    
    # Time period
    interval: str = Field(..., regex="^[1-9][0-9]*[smhd]$")
    period_start: datetime
    period_end: datetime
    
    @validator('high')
    def validate_high(cls, high, values):
        """Ensure high is highest price"""
        if 'low' in values and high < values['low']:
            raise ValueError(f"High {high} must be >= low {values['low']}")
        return high
    
    @validator('low')
    def validate_low(cls, low, values):
        """Ensure low is lowest price"""
        for field in ['open', 'close']:
            if field in values and low > values[field]:
                raise ValueError(f"Low {low} must be <= {field} {values[field]}")
        return low

class OrderBookLevel(BaseModel):
    """Single order book level"""
    price: float = Field(..., gt=0)
    quantity: int = Field(..., ge=0)
    order_count: Optional[int] = Field(None, ge=0)

class OrderBook(MarketData):
    """Order book/Level 2 market data"""
    
    data_type: MarketDataType = MarketDataType.ORDER_BOOK
    
    # Order book levels
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    
    # Metadata
    depth: int = Field(..., gt=0)
    is_snapshot: bool = Field(default=True)
    
    @validator('bids')
    def validate_bid_ordering(cls, bids):
        """Ensure bids are in descending order"""
        prices = [level.price for level in bids]
        if prices != sorted(prices, reverse=True):
            raise ValueError("Bids must be in descending price order")
        return bids
    
    @validator('asks')
    def validate_ask_ordering(cls, asks):
        """Ensure asks are in ascending order"""
        prices = [level.price for level in asks]
        if prices != sorted(prices):
            raise ValueError("Asks must be in ascending price order")
        return asks

class MarketDataBatch(BaseModel):
    """Batch of market data messages"""
    
    batch_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    messages: List[Dict] = Field(...)
    count: int = Field(...)
    
    @validator('count')
    def validate_count(cls, count, values):
        """Ensure count matches messages"""
        if 'messages' in values and count != len(values['messages']):
            raise ValueError(f"Count {count} doesn't match messages {len(values['messages'])}")
        return count
