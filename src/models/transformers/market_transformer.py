import torch
import torch.nn as nn
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0)]

class MarketTransformer(nn.Module):
    def __init__(self, 
                 input_dim: int = 10,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.price_predictor = nn.Linear(d_model, 1)
        self.volatility_predictor = nn.Linear(d_model, 1)
        self.action_classifier = nn.Linear(d_model, 3)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        last_hidden = x[-1]
        
        price_pred = self.price_predictor(last_hidden)
        volatility_pred = torch.sigmoid(self.volatility_predictor(last_hidden))
        action_logits = self.action_classifier(last_hidden)
        
        return {
            "price": price_pred,
            "volatility": volatility_pred,
            "action": torch.softmax(action_logits, dim=-1)
        }

class TransformerTrader:
    def __init__(self, input_dim: int = 10, seq_len: int = 50):
        self.model = MarketTransformer(input_dim=input_dim)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.seq_len = seq_len
        self.memory = []
        
    def predict(self, market_data: np.ndarray):
        if len(market_data) < self.seq_len:
            return None
            
        sequence = torch.FloatTensor(market_data[-self.seq_len:]).unsqueeze(1)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(sequence)
        
        return {
            "predicted_price": predictions["price"].item(),
            "predicted_volatility": predictions["volatility"].item(),
            "action": predictions["action"].argmax().item(),
            "action_probs": predictions["action"].squeeze().numpy()
        }
    
    def train_step(self, sequences, targets):
        self.model.train()
        
        sequences = torch.FloatTensor(sequences)
        target_prices = torch.FloatTensor(targets["prices"])
        target_actions = torch.LongTensor(targets["actions"])
        
        predictions = self.model(sequences)
        
        price_loss = nn.MSELoss()(predictions["price"].squeeze(), target_prices)
        action_loss = nn.CrossEntropyLoss()(predictions["action"], target_actions)
        
        total_loss = price_loss + action_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()

