"""Kafka Consumer for Market Data Ingestion"""
import asyncio
import json
import signal
import sys
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from confluent_kafka import Consumer, KafkaError, Message
from confluent_kafka.admin import AdminClient
import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import orjson

from ..kafka_config import kafka_config
from ..schemas.market_data import (
    MarketData, Quote, Trade, OHLCV, OrderBook, MarketDataBatch
)

# Configure structured logging
logger = structlog.get_logger()

# Prometheus metrics
MESSAGES_CONSUMED = Counter(
    'kafka_messages_consumed_total',
    'Total messages consumed from Kafka',
    ['topic', 'data_type']
)

MESSAGES_FAILED = Counter(
    'kafka_messages_failed_total',
    'Total messages failed to process',
    ['topic', 'error_type']
)

PROCESSING_TIME = Histogram(
    'kafka_message_processing_seconds',
    'Time spent processing messages',
    ['topic', 'data_type']
)

LAG_GAUGE = Gauge(
    'kafka_consumer_lag',
    'Current consumer lag',
    ['topic', 'partition']
)

CONSUMER_STATUS = Gauge(
    'kafka_consumer_status',
    'Consumer status (1=running, 0=stopped)'
)

class MarketDataConsumer:
    """High-performance Kafka consumer for market data"""
    
    def __init__(
        self,
        topics: Optional[List[str]] = None,
        message_handler: Optional[Callable] = None
    ):
        """
        Initialize Kafka consumer
        
        Args:
            topics: List of topics to subscribe to
            message_handler: Callback for processing messages
        """
        self.config = kafka_config
        self.consumer = None
        self.running = False
        self.message_count = 0
        self.error_count = 0
        
        # Topics to consume
        self.topics = topics or [
            self.config.market_data_topic,
            self.config.fundamental_data_topic,
            self.config.alternative_data_topic
        ]
        
        # Message handler
        self.message_handler = message_handler or self.default_handler
        
        # Start metrics server if enabled
        if self.config.enable_metrics:
            start_http_server(self.config.metrics_port)
            logger.info(
                "Started metrics server",
                port=self.config.metrics_port
            )
    
    def connect(self) -> bool:
        """Connect to Kafka cluster"""
        try:
            # Create consumer
            config = self.config.get_consumer_config()
            self.consumer = Consumer(config)
            
            # Subscribe to topics
            self.consumer.subscribe(self.topics)
            
            logger.info(
                "Connected to Kafka",
                bootstrap_servers=self.config.bootstrap_servers,
                topics=self.topics,
                consumer_group=self.config.consumer_group
            )
            
            CONSUMER_STATUS.set(1)
            return True
            
        except Exception as e:
            logger.error(
                "Failed to connect to Kafka",
                error=str(e),
                bootstrap_servers=self.config.bootstrap_servers
            )
            CONSUMER_STATUS.set(0)
            return False
    
    def start(self):
        """Start consuming messages"""
        if not self.consumer:
            if not self.connect():
                raise RuntimeError("Failed to connect to Kafka")
        
        self.running = True
        logger.info("Starting consumer loop")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            while self.running:
                # Poll for messages
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    self._handle_error(msg)
                else:
                    self._process_message(msg)
                    
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        finally:
            self.shutdown()
    
    def _process_message(self, msg: Message):
        """Process a single message"""
        topic = msg.topic()
        partition = msg.partition()
        offset = msg.offset()
        
        with PROCESSING_TIME.labels(
            topic=topic,
            data_type='unknown'
        ).time():
            try:
                # Decode message
                key = msg.key().decode('utf-8') if msg.key() else None
                value = orjson.loads(msg.value())
                
                # Parse into schema
                data = self._parse_message(value)
                data_type = data.data_type if hasattr(data, 'data_type') else 'unknown'
                
                # Call handler
                self.message_handler(data, key, topic)
                
                # Update metrics
                MESSAGES_CONSUMED.labels(
                    topic=topic,
                    data_type=data_type
                ).inc()
                
                self.message_count += 1
                
                # Log progress every 1000 messages
                if self.message_count % 1000 == 0:
                    logger.info(
                        "Processing progress",
                        messages_processed=self.message_count,
                        errors=self.error_count,
                        current_offset=offset,
                        topic=topic,
                        partition=partition
                    )
                
            except Exception as e:
                logger.error(
                    "Failed to process message",
                    error=str(e),
                    topic=topic,
                    partition=partition,
                    offset=offset
                )
                
                MESSAGES_FAILED.labels(
                    topic=topic,
                    error_type=type(e).__name__
                ).inc()
                
                self.error_count += 1
    
    def _parse_message(self, value: Dict) -> MarketData:
        """Parse message into appropriate schema"""
        data_type = value.get('data_type', 'quote')
        
        schema_map = {
            'quote': Quote,
            'trade': Trade,
            'ohlcv': OHLCV,
            'order_book': OrderBook
        }
        
        schema_class = schema_map.get(data_type, MarketData)
        return schema_class(**value)
    
    def _handle_error(self, msg: Message):
        """Handle Kafka errors"""
        error = msg.error()
        
        if error.code() == KafkaError._PARTITION_EOF:
            # End of partition - not really an error
            logger.debug(
                "Reached end of partition",
                topic=msg.topic(),
                partition=msg.partition()
            )
        else:
            logger.error(
                "Kafka error",
                error=str(error),
                code=error.code()
            )
            
            MESSAGES_FAILED.labels(
                topic=msg.topic() if msg.topic() else 'unknown',
                error_type='kafka_error'
            ).inc()
    
    def default_handler(self, data: MarketData, key: str, topic: str):
        """Default message handler - just logs"""
        logger.info(
            "Received market data",
            symbol=data.symbol,
            exchange=data.exchange,
            data_type=data.data_type,
            timestamp=data.timestamp.isoformat(),
            topic=topic,
            key=key
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def shutdown(self):
        """Clean shutdown"""
        if self.consumer:
            logger.info("Closing consumer...")
            self.consumer.close()
            CONSUMER_STATUS.set(0)
            
        logger.info(
            "Consumer shutdown complete",
            total_messages=self.message_count,
            total_errors=self.error_count
        )
    
    async def start_async(self):
        """Async version of start method"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.start)

class BatchedMarketDataConsumer(MarketDataConsumer):
    """Consumer that batches messages for efficient processing"""
    
    def __init__(
        self,
        topics: Optional[List[str]] = None,
        batch_size: int = 100,
        batch_timeout: float = 1.0,
        batch_handler: Optional[Callable] = None
    ):
        super().__init__(topics=topics)
        
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.batch_handler = batch_handler or self.default_batch_handler
        self.current_batch = []
        self.last_batch_time = datetime.utcnow()
    
    def _process_message(self, msg: Message):
        """Process message and add to batch"""
        try:
            # Parse message
            value = orjson.loads(msg.value())
            data = self._parse_message(value)
            
            # Add to batch
            self.current_batch.append(data)
            
            # Check if batch is ready
            if self._should_flush_batch():
                self._flush_batch()
                
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            self.error_count += 1
    
    def _should_flush_batch(self) -> bool:
        """Check if batch should be flushed"""
        if len(self.current_batch) >= self.batch_size:
            return True
        
        elapsed = (datetime.utcnow() - self.last_batch_time).total_seconds()
        if elapsed >= self.batch_timeout and self.current_batch:
            return True
        
        return False
    
    def _flush_batch(self):
        """Flush current batch"""
        if not self.current_batch:
            return
        
        try:
            batch = MarketDataBatch(
                batch_id=f"batch_{self.message_count}",
                timestamp=datetime.utcnow(),
                messages=[msg.dict() for msg in self.current_batch],
                count=len(self.current_batch)
            )
            
            self.batch_handler(batch)
            
            logger.info(
                "Flushed batch",
                batch_size=len(self.current_batch),
                batch_id=batch.batch_id
            )
            
            self.message_count += len(self.current_batch)
            self.current_batch = []
            self.last_batch_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to flush batch: {e}")
            self.error_count += 1
    
    def default_batch_handler(self, batch: MarketDataBatch):
        """Default batch handler"""
        logger.info(
            "Processed batch",
            batch_id=batch.batch_id,
            message_count=batch.count
        )
    
    def shutdown(self):
        """Flush remaining messages before shutdown"""
        if self.current_batch:
            self._flush_batch()
        super().shutdown()
