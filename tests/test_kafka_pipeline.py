"""Test Kafka Data Ingestion"""
import asyncio
import json
import random
from datetime import datetime
from confluent_kafka import Producer
import time

def create_test_producer():
    """Create Kafka producer for testing"""
    config = {
        'bootstrap.servers': 'localhost:9093',
        'client.id': 'test-producer'
    }
    return Producer(config)

def generate_test_quote():
    """Generate test quote data"""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    symbol = random.choice(symbols)
    
    base_price = {
        'AAPL': 180,
        'GOOGL': 140,
        'MSFT': 380,
        'AMZN': 175,
        'TSLA': 250
    }[symbol]
    
    # Add some randomness
    bid = base_price * (1 + random.uniform(-0.01, 0.01))
    ask = bid * (1 + random.uniform(0.0001, 0.001))
    
    return {
        'data_type': 'quote',
        'symbol': symbol,
        'exchange': 'NASDAQ',
        'asset_class': 'equity',
        'timestamp': datetime.utcnow().isoformat(),
        'bid': round(bid, 2),
        'ask': round(ask, 2),
        'bid_size': random.randint(100, 1000),
        'ask_size': random.randint(100, 1000),
        'last': round(bid + (ask - bid) / 2, 2),
        'volume': random.randint(1000000, 10000000)
    }

def test_producer():
    """Send test messages to Kafka"""
    producer = create_test_producer()
    
    print("Sending test messages to Kafka...")
    
    for i in range(100):
        # Generate test data
        quote = generate_test_quote()
        
        # Send to Kafka
        producer.produce(
            topic='market-data',
            key=quote['symbol'],
            value=json.dumps(quote).encode('utf-8')
        )
        
        if i % 10 == 0:
            print(f"Sent {i} messages")
            producer.flush()
        
        time.sleep(0.1)
    
    producer.flush()
    print("Test complete!")

async def test_consumer():
    """Test the consumer"""
    from src.data_pipeline.consumers.market_data_consumer import MarketDataConsumer
    from src.data_pipeline.feature_pipeline import RealTimeFeatureEngine
    
    print("Starting feature engine...")
    engine = RealTimeFeatureEngine()
    
    # Run for 30 seconds
    try:
        await asyncio.wait_for(engine.start(), timeout=30)
    except asyncio.TimeoutError:
        print("Test completed after 30 seconds")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "produce":
        test_producer()
    elif len(sys.argv) > 1 and sys.argv[1] == "consume":
        asyncio.run(test_consumer())
    else:
        print("Usage: python test_kafka.py [produce|consume]")

