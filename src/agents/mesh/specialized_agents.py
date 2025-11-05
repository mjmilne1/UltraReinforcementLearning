import sys
sys.path.insert(0, '.')
from src.agents.mesh.agent_framework import BaseAgent, AgentRole, AgentMessage
from src.market_data import MarketDataFetcher
from src.indicators.technical_indicators import TechnicalIndicators
import numpy as np
import asyncio

class MarketAnalyzerAgent(BaseAgent):
    '''Analyzes market conditions and opportunities'''
    
    def __init__(self):
        super().__init__("MarketAnalyzer", AgentRole.MARKET_ANALYZER)
        self.fetcher = MarketDataFetcher()
        self.indicators = TechnicalIndicators()
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
        
    async def think(self):
        '''Analyze markets and identify opportunities'''
        for symbol in self.symbols:
            try:
                # Get current price
                data = self.fetcher.get_live_price(symbol)
                
                if data:
                    # Analyze conditions
                    analysis = {
                        'symbol': symbol,
                        'price': data.get('price', 0),
                        'change_pct': data.get('change_pct', 0),
                        'volume': data.get('volume', 0),
                        'trend': 'bullish' if data.get('change_pct', 0) > 0 else 'bearish',
                        'strength': abs(data.get('change_pct', 0))
                    }
                    
                    # Send to strategy selector
                    await self.send_message(
                        "StrategySelector",
                        "market_analysis",
                        analysis
                    )
                    
                    # Alert if significant move
                    if abs(data.get('change_pct', 0)) > 2:
                        await self.send_message(
                            "AlertCoordinator",
                            "significant_move",
                            analysis
                        )
                        
            except Exception as e:
                pass
        
        await asyncio.sleep(30)  # Analyze every 30 seconds

class RiskManagerAgent(BaseAgent):
    '''Manages risk and position limits'''
    
    def __init__(self):
        super().__init__("RiskManager", AgentRole.RISK_MANAGER)
        self.risk_limits = {
            'max_position_size': 0.2,
            'max_drawdown': 0.1,
            'max_daily_loss': 0.05,
            'max_correlation': 0.7
        }
        
    async def handle_message(self, message: AgentMessage):
        '''Process risk-related messages'''
        if message.message_type == "position_request":
            # Evaluate position request
            approved = await self.evaluate_risk(message.payload)
            
            await self.send_message(
                message.sender,
                "risk_decision",
                {'approved': approved, 'original_request': message.payload}
            )
    
    async def evaluate_risk(self, request: dict) -> bool:
        '''Evaluate if a trade meets risk criteria'''
        position_size = request.get('position_size', 0)
        
        # Check position size limit
        if position_size > self.risk_limits['max_position_size']:
            return False
        
        # More risk checks here...
        return True

class StrategySelectorAgent(BaseAgent):
    '''Selects optimal strategy based on conditions'''
    
    def __init__(self):
        super().__init__("StrategySelector", AgentRole.STRATEGY_SELECTOR)
        self.market_conditions = {}
        self.strategy_performance = {
            'momentum': 0.57,  # Historical performance
            'mean_reversion': 0.25,
            'dqn': 0.34,
            'ensemble': 0.40
        }
        
    async def handle_message(self, message: AgentMessage):
        '''Process strategy-related messages'''
        if message.message_type == "market_analysis":
            # Update market conditions
            symbol = message.payload.get('symbol')
            self.market_conditions[symbol] = message.payload
            
            # Select strategy
            strategy = self.select_strategy(message.payload)
            
            # Send to position manager
            await self.send_message(
                "PositionManager",
                "strategy_signal",
                {
                    'symbol': symbol,
                    'strategy': strategy,
                    'conditions': message.payload
                }
            )
    
    def select_strategy(self, market_data: dict) -> str:
        '''Select best strategy for current conditions'''
        trend = market_data.get('trend')
        strength = market_data.get('strength', 0)
        
        if trend == 'bullish' and strength > 1:
            return 'momentum'
        elif abs(strength) < 0.5:
            return 'mean_reversion'
        else:
            return 'ensemble'

class PositionManagerAgent(BaseAgent):
    '''Manages positions and executes trades'''
    
    def __init__(self):
        super().__init__("PositionManager", AgentRole.POSITION_MANAGER)
        self.positions = {}
        self.pending_trades = []
        
    async def handle_message(self, message: AgentMessage):
        '''Process position-related messages'''
        if message.message_type == "strategy_signal":
            # Request risk approval
            await self.send_message(
                "RiskManager",
                "position_request",
                {
                    'symbol': message.payload['symbol'],
                    'position_size': 0.1,  # 10% of portfolio
                    'strategy': message.payload['strategy']
                }
            )
        
        elif message.message_type == "risk_decision":
            if message.payload['approved']:
                # Execute trade
                await self.send_message(
                    "Executor",
                    "execute_trade",
                    message.payload['original_request']
                )

class ExecutorAgent(BaseAgent):
    '''Executes approved trades'''
    
    def __init__(self):
        super().__init__("Executor", AgentRole.EXECUTOR)
        self.trade_count = 0
        
    async def handle_message(self, message: AgentMessage):
        '''Execute trades'''
        if message.message_type == "execute_trade":
            symbol = message.payload['symbol']
            
            # Simulate trade execution
            self.trade_count += 1
            
            result = {
                'trade_id': self.trade_count,
                'symbol': symbol,
                'status': 'executed',
                'timestamp': message.timestamp
            }
            
            # Notify all agents
            await self.send_message(
                "BROADCAST",
                "trade_executed",
                result
            )
            
            print(f"🔄 Trade #{self.trade_count}: {symbol} executed")

class AlertCoordinatorAgent(BaseAgent):
    '''Coordinates alerts across the mesh'''
    
    def __init__(self):
        super().__init__("AlertCoordinator", AgentRole.ALERT_COORDINATOR)
        
    async def handle_message(self, message: AgentMessage):
        '''Process alert requests'''
        if message.message_type == "significant_move":
            symbol = message.payload['symbol']
            change = message.payload['change_pct']
            print(f"🔔 ALERT: {symbol} moved {change:+.2f}%")
        
        elif message.message_type == "trade_executed":
            print(f"✅ Trade executed: {message.payload}")
