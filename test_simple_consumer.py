from confluent_kafka import Consumer, KafkaError
import json

def test_consumer():
    """Simple test consumer"""
    
    # Configure consumer
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'test-consumer-group',
        'auto.offset.reset': 'earliest'
    }
    
    # Create consumer
    consumer = Consumer(conf)
    consumer.subscribe(['market-data'])
    
    print("Starting consumer... Press Ctrl+C to stop")
    print("-" * 50)
    
    message_count = 0
    
    try:
        while True:
            # Poll for messages
            msg = consumer.poll(1.0)
            
            if msg is None:
                continue
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    print(f"Reached end of partition {msg.partition()}")
                else:
                    print(f"Error: {msg.error()}")
            else:
                # Process message
                key = msg.key().decode('utf-8') if msg.key() else None
                value = json.loads(msg.value().decode('utf-8'))
                
                message_count += 1
                
                print(f"Message #{message_count}")
                print(f"  Topic: {msg.topic()}")
                print(f"  Key: {key}")
                print(f"  Symbol: {value.get('symbol')}")
                print(f"  Price: ${value.get('last', 0):.2f}")
                print(f"  Timestamp: {value.get('timestamp')}")
                print("-" * 50)
                
    except KeyboardInterrupt:
        print(f"\nShutting down... Processed {message_count} messages")
    finally:
        consumer.close()

if __name__ == "__main__":
    test_consumer()
