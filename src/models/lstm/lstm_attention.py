import torch
import torch.nn as nn
import numpy as np

class AttentionLayer(nn.Module):
    '''Attention mechanism for LSTM'''
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        weighted = torch.bmm(attention_weights.transpose(1, 2), lstm_output)
        return weighted.squeeze(1), attention_weights

class LSTMAttentionTrader(nn.Module):
    '''LSTM with attention for trading'''
    def __init__(self, 
                 input_dim: int = 10,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim * 2)
        
        # Prediction heads
        self.price_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.direction_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # Up/Down/Flat
        )
        
        self.confidence_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attended, attention_weights = self.attention(lstm_out)
        
        # Predictions
        price_pred = self.price_predictor(attended)
        direction = self.direction_classifier(attended)
        confidence = self.confidence_scorer(attended)
        
        return {
            'price': price_pred,
            'direction': torch.softmax(direction, dim=-1),
            'confidence': confidence,
            'attention': attention_weights
        }

class LSTMPredictor:
    '''LSTM-based price prediction and trading'''
    def __init__(self, input_dim: int = 10):
        self.model = LSTMAttentionTrader(input_dim=input_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scaler = None
        
    def prepare_sequences(self, data: np.ndarray, seq_len: int = 30):
        '''Prepare sequences for LSTM'''
        sequences = []
        targets = []
        
        for i in range(len(data) - seq_len - 1):
            seq = data[i:i+seq_len]
            target = data[i+seq_len]
            sequences.append(seq)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
    
    def predict_next(self, sequence: np.ndarray):
        '''Predict next price movement'''
        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(seq_tensor)
        
        direction_probs = predictions['direction'][0].numpy()
        confidence = predictions['confidence'][0].item()
        
        # Trading decision based on predictions
        if confidence > 0.7:
            if direction_probs[0] > 0.5:  # Up
                action = 'BUY'
            elif direction_probs[1] > 0.5:  # Down
                action = 'SELL'
            else:
                action = 'HOLD'
        else:
            action = 'HOLD'  # Low confidence
            
        return {
            'action': action,
            'confidence': confidence,
            'up_prob': direction_probs[0],
            'down_prob': direction_probs[1],
            'flat_prob': direction_probs[2]
        }
