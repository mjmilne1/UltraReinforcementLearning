"""Kafka Configuration for Ultra RL Platform"""
from pydantic import BaseSettings, Field
from typing import List, Optional
import os

class KafkaConfig(BaseSettings):
    """Kafka consumer configuration"""
    
    # Connection settings
    bootstrap_servers: str = Field(
        default="localhost:9093",
        env="KAFKA_BOOTSTRAP_SERVERS"
    )
    
    # Consumer settings
    consumer_group: str = Field(
        default="ultra-rl-consumer",
        env="KAFKA_CONSUMER_GROUP"
    )
    
    # Topics
    market_data_topic: str = Field(
        default="market-data",
        env="KAFKA_MARKET_DATA_TOPIC"
    )
    
    fundamental_data_topic: str = Field(
        default="fundamental-data",
        env="KAFKA_FUNDAMENTAL_TOPIC"
    )
    
    alternative_data_topic: str = Field(
        default="alternative-data",
        env="KAFKA_ALTERNATIVE_TOPIC"
    )
    
    # Performance settings
    max_poll_records: int = Field(
        default=500,
        env="KAFKA_MAX_POLL_RECORDS"
    )
    
    fetch_min_bytes: int = Field(
        default=1,
        env="KAFKA_FETCH_MIN_BYTES"
    )
    
    fetch_max_wait_ms: int = Field(
        default=500,
        env="KAFKA_FETCH_MAX_WAIT_MS"
    )
    
    # Security
    security_protocol: str = Field(
        default="PLAINTEXT",
        env="KAFKA_SECURITY_PROTOCOL"
    )
    
    sasl_mechanism: Optional[str] = Field(
        default=None,
        env="KAFKA_SASL_MECHANISM"
    )
    
    sasl_username: Optional[str] = Field(
        default=None,
        env="KAFKA_SASL_USERNAME"
    )
    
    sasl_password: Optional[str] = Field(
        default=None,
        env="KAFKA_SASL_PASSWORD"
    )
    
    # Monitoring
    enable_metrics: bool = Field(
        default=True,
        env="KAFKA_ENABLE_METRICS"
    )
    
    metrics_port: int = Field(
        default=9090,
        env="KAFKA_METRICS_PORT"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def get_consumer_config(self) -> dict:
        """Get Kafka consumer configuration dictionary"""
        config = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.consumer_group,
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True,
            'auto.commit.interval.ms': 5000,
            'session.timeout.ms': 30000,
            'max.poll.records': self.max_poll_records,
            'fetch.min.bytes': self.fetch_min_bytes,
            'fetch.max.wait.ms': self.fetch_max_wait_ms,
            'security.protocol': self.security_protocol,
        }
        
        # Add authentication if configured
        if self.sasl_mechanism:
            config.update({
                'sasl.mechanism': self.sasl_mechanism,
                'sasl.username': self.sasl_username,
                'sasl.password': self.sasl_password,
            })
        
        return config

# Global config instance
kafka_config = KafkaConfig()

