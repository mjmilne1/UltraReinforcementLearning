from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
import asyncio
import json
from datetime import datetime
import numpy as np

class AgentRole(Enum):
    '''Specialized agent roles in the mesh'''
    MARKET_ANALYZER = "market_analyzer"
    RISK_MANAGER = "risk_manager"
    STRATEGY_SELECTOR = "strategy_selector"
    POSITION_MANAGER = "position_manager"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    ALERT_COORDINATOR = "alert_coordinator"
    EXECUTOR = "executor"

@dataclass
class AgentMessage:
    '''Message structure for inter-agent communication'''
    sender: str
    recipient: str
    message_type: str
    payload: Dict
    timestamp: datetime
    priority: int = 0

class BaseAgent:
    '''Base class for all specialized agents'''
    
    def __init__(self, name: str, role: AgentRole):
        self.name = name
        self.role = role
        self.inbox = asyncio.Queue()
        self.outbox = asyncio.Queue()
        self.state = {}
        self.running = False
        self.decision_history = []
        
    async def receive_message(self, message: AgentMessage):
        '''Receive message from another agent'''
        await self.inbox.put(message)
    
    async def send_message(self, recipient: str, message_type: str, payload: Dict):
        '''Send message to another agent'''
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now()
        )
        await self.outbox.put(message)
    
    async def process_messages(self):
        '''Process incoming messages'''
        while not self.inbox.empty():
            message = await self.inbox.get()
            await self.handle_message(message)
    
    async def handle_message(self, message: AgentMessage):
        '''Override in specialized agents'''
        pass
    
    async def think(self):
        '''Agent's main decision-making loop'''
        pass
    
    async def run(self):
        '''Main agent execution loop'''
        self.running = True
        while self.running:
            await self.process_messages()
            await self.think()
            await asyncio.sleep(1)  # Prevent CPU spinning

class AgenticMesh:
    '''Coordinator for multiple# Create the base agent and mesh coordinator
@"
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
import asyncio
import json
from datetime import datetime
import numpy as np

class AgentRole(Enum):
    '''Specialized agent roles in the mesh'''
    MARKET_ANALYZER = "market_analyzer"
    RISK_MANAGER = "risk_manager"
    STRATEGY_SELECTOR = "strategy_selector"
    POSITION_MANAGER = "position_manager"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    ALERT_COORDINATOR = "alert_coordinator"
    EXECUTOR = "executor"

@dataclass
class AgentMessage:
    '''Message structure for inter-agent communication'''
    sender: str
    recipient: str
    message_type: str
    payload: Dict
    timestamp: datetime
    priority: int = 0

class BaseAgent:
    '''Base class for all specialized agents'''
    
    def __init__(self, name: str, role: AgentRole):
        self.name = name
        self.role = role
        self.inbox = asyncio.Queue()
        self.outbox = asyncio.Queue()
        self.state = {}
        self.running = False
        self.decision_history = []
        
    async def receive_message(self, message: AgentMessage):
        '''Receive message from another agent'''
        await self.inbox.put(message)
    
    async def send_message(self, recipient: str, message_type: str, payload: Dict):
        '''Send message to another agent'''
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now()
        )
        await self.outbox.put(message)
    
    async def process_messages(self):
        '''Process incoming messages'''
        while not self.inbox.empty():
            message = await self.inbox.get()
            await self.handle_message(message)
    
    async def handle_message(self, message: AgentMessage):
        '''Override in specialized agents'''
        pass
    
    async def think(self):
        '''Agent's main decision-making loop'''
        pass
    
    async def run(self):
        '''Main agent execution loop'''
        self.running = True
        while self.running:
            await self.process_messages()
            await self.think()
            await asyncio.sleep(1)  # Prevent CPU spinning

class AgenticMesh:
    '''Coordinator for multiple autonomous agents'''
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_bus = asyncio.Queue()
        self.running = False
        self.mesh_state = {
            'market_conditions': None,
            'active_strategies': [],
            'current_positions': {},
            'risk_metrics': {},
            'performance': {}
        }
        
    def register_agent(self, agent: BaseAgent):
        '''Register an agent in the mesh'''
        self.agents[agent.name] = agent
        print(f"✅ Registered agent: {agent.name} ({agent.role.value})")
    
    async def route_messages(self):
        '''Route messages between agents'''
        while self.running:
            # Collect messages from all agents
            for agent in self.agents.values():
                while not agent.outbox.empty():
                    message = await agent.outbox.get()
                    
                    # Route to recipient
                    if message.recipient in self.agents:
                        await self.agents[message.recipient].receive_message(message)
                    elif message.recipient == "BROADCAST":
                        # Broadcast to all agents
                        for other_agent in self.agents.values():
                            if other_agent.name != message.sender:
                                await other_agent.receive_message(message)
            
            await asyncio.sleep(0.1)
    
    async def run(self):
        '''Run the agentic mesh'''
        self.running = True
        print("\n🚀 Starting Agentic Mesh...")
        print("="*50)
        
        # Start all agents
        tasks = []
        for agent in self.agents.values():
            tasks.append(asyncio.create_task(agent.run()))
        
        # Start message router
        tasks.append(asyncio.create_task(self.route_messages()))
        
        # Run until interrupted
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\n⏹️ Stopping agentic mesh...")
            self.running = False
            for agent in self.agents.values():
                agent.running = False
