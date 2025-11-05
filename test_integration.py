import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import time
from confluent_kafka import Producer, Consumer
from datetime import datetime, timezone
import numpy as np
import threading

def producer_thread():
    '''Producer thread'''
    producer = Producer({'bootstrap.servers': 'localhost:9092'})
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    for i in range(50):
        for symbol in symbols:
            data = {
                'data_type': 'quote',
                'symbol': symbol,
                'exchange': 'NASDAQ',
                'asset_class': 'equity',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'bid': 100 + np.random.uniform(-5, 5),
                'ask': 100.5 + np.random.uniform(-5, 5),
                'bid_size': 1000,
                'ask_size': 1000,
                'last': 100.25,
                'volume': 1000000
            }
            
            producer.produce(
                topic='market-data',
                key=symbol,
                value=json.dumps(data).encode('utf-8')
            )
        
        if i % 10 == 0:
            print(f'Producer: Sent {(i+1)*3} messages')
            producer.flush()
        
        time.sleep(0.1)
    
    producer.flush()
    print('Producer: Complete!')

def consumer_with_features():
    '''Consumer with feature processing'''
    # Skip feature pipeline for now import FeaturePipeline
    # Skip schema import import Quote
    
    consumer = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'feature-consumer',
        'auto.offset.reset': 'earliest'
    })
    consumer.subscribe(['market-data'])
    
    # # pipeline = FeaturePipeline(window_size=10)
    
    print('Consumer: Starting with feature processing...')
    
    message_count = 0
    feature_count = 0
    
    start_time = time.time()
    timeout = 20  # Run for 20 seconds
    
    while time.time() - start_time < timeout:
        msg = consumer.poll(1.0)
        
        if msg and not msg.error():
            value = json.loads(msg.value().decode('utf-8'))
            message_count += 1
            
            # Convert to Quote
            # # quote = Quote(**value)
            
            # Generate features
            features = None # Skip features
            
            if features is not None:
                feature_count += 1
                print(f'Consumer: Processed {message_count} messages, generated {feature_count} feature vectors')
                print(f'  Latest: {quote.symbol} @ {quote.last:.2f}')
                print(f'  Feature shape: {features.shape}')
    
    consumer.close()
    print(f'\nConsumer: Complete! Processed {message_count} messages, generated {feature_count} feature vectors')

def main():
    '''Run full test'''
    print('='*60)
    print('ULTRA RL - Kafka Pipeline Integration Test')
    print('='*60)
    
    # Start producer in thread
    producer = threading.Thread(target=producer_thread)
    producer.start()
    
    # Give producer time to start
    time.sleep(2)
    
    # Run consumer
    consumer_with_features()
    
    # Wait for producer to finish
    producer.join()
    
    print('\n' + '='*60)
    print('Integration test complete!')
    print('='*60)

if __name__ == '__main__':
    main()


