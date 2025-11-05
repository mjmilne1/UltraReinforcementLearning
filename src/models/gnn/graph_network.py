import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphAttentionLayer(nn.Module):
    '''Graph Attention Network layer'''
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
    def forward(self, x, adj):
        h = self.W(x)
        N = h.size(0)
        
        # Attention mechanism
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1),
                           h.repeat(N, 1)], dim=1).view(N, N, 2 * h.size(1))
        e = F.leaky_relu(self.a(a_input).squeeze(2))
        
        # Masked attention
        attention = torch.where(adj > 0, e, torch.full_like(e, -1e9))
        attention = F.softmax(attention, dim=1)
        
        return torch.matmul(attention, h), attention

class MarketGraphNetwork(nn.Module):
    '''GNN for analyzing market correlations'''
    def __init__(self, 
                 num_assets: int,
                 feature_dim: int = 10,
                 hidden_dim: int = 64):
        super().__init__()
        
        # Graph layers
        self.gat1 = GraphAttentionLayer(feature_dim, hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim)
        
        # Asset-specific predictors
        self.asset_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 3)  # Buy/Hold/Sell per asset
            ) for _ in range(num_assets)
        ])
        
        # Portfolio optimizer
        self.portfolio_optimizer = nn.Sequential(
            nn.Linear(hidden_dim * num_assets, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_assets),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x, correlation_matrix):
        # Build adjacency matrix from correlations
        adj = (torch.abs(correlation_matrix) > 0.3).float()
        
        # Graph convolutions
        h1, att1 = self.gat1(x, adj)
        h1 = F.relu(h1)
        h2, att2 = self.gat2(h1, adj)
        
        # Asset predictions
        asset_actions = []
        for i, predictor in enumerate(self.asset_predictor):
            action = predictor(h2[i])
            asset_actions.append(action)
        
        # Portfolio weights
        combined = h2.flatten()
        weights = self.portfolio_optimizer(combined)
        
        return {
            'actions': torch.stack(asset_actions),
            'weights': weights,
            'attention': att2
        }

class CorrelationTrader:
    '''Trade based on asset correlations using GNN'''
    def __init__(self, assets: list):
        self.assets = assets
        self.num_assets = len(assets)
        self.model = MarketGraphNetwork(
            num_assets=self.num_assets,
            feature_dim=10
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def calculate_correlations(self, returns_data: dict):
        '''Calculate correlation matrix'''
        returns = []
        for asset in self.assets:
            if asset in returns_data:
                returns.append(returns_data[asset])
        
        if not returns:
            return None
            
        returns_array = np.array(returns)
        correlation = np.corrcoef(returns_array)
        return torch.FloatTensor(correlation)
    
    def get_portfolio_allocation(self, features: np.ndarray, returns_data: dict):
        '''Get optimal portfolio allocation'''
        correlation = self.calculate_correlations(returns_data)
        if correlation is None:
            return None
            
        features_tensor = torch.FloatTensor(features)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features_tensor, correlation)
        
        allocation = {}
        actions = predictions['actions'].argmax(dim=1)
        weights = predictions['weights'].numpy()
        
        for i, asset in enumerate(self.assets):
            allocation[asset] = {
                'weight': weights[i],
                'action': ['HOLD', 'BUY', 'SELL'][actions[i].item()]
            }
            
        return allocation
