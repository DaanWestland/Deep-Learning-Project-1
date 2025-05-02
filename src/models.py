"""
This module contains the neural network model definitions for time series forecasting.
The models are implemented using PyTorch's nn.Module framework.
"""

import torch.nn as nn

class GRUForecast(nn.Module):
    """
    A Gated Recurrent Unit (GRU) based model for time series forecasting.
    
    This model uses a single-layer GRU followed by a linear layer to predict the next value
    in a time series sequence. The GRU architecture is chosen for its ability to capture
    temporal dependencies while being computationally more efficient than LSTM.
    
    Attributes:
        lookback (int): Number of previous time steps used for prediction
        gru (nn.GRU): The GRU layer that processes the input sequence
        fc (nn.Linear): Final fully connected layer that produces the prediction
    """
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.2, lookback=100):
        """
        Initialize the GRUForecast model.
        
        Args:
            input_size (int): Number of features in the input time series
            hidden_size (int): Number of features in the hidden state
            num_layers (int): Number of GRU layers
            dropout (float): Dropout probability between GRU layers (if num_layers > 1)
            lookback (int): Number of previous time steps to consider for prediction
        """
        super().__init__()
        self.lookback = lookback  # Store lookback window size for reference
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,  # Input format: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0.0  # Only apply dropout if multiple layers
        )
        self.fc = nn.Linear(hidden_size, 1)  # Final layer to produce single prediction

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            torch.Tensor: Predicted value of shape (batch_size, 1)
        """
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)         # Process sequence through GRU: out shape (batch, seq_len, hidden_size)
        last = out[:, -1, :]         # Extract the last hidden state for prediction
        return self.fc(last)         # Transform hidden state to prediction: (batch, 1)
