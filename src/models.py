"""
This module contains the neural network model definitions for time series forecasting.
The models are implemented using PyTorch's nn.Module framework.
Supports both single-step and multi-step (chunked/MIMO) prediction via a configurable horizon.
"""
import torch
import torch.nn as nn

class GRUForecast(nn.Module):
    """
    A Gated Recurrent Unit (GRU) based model for time series forecasting,
    supporting multi-step output (horizon).
    """
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        horizon: int = 1
    ):
        """
        Initialize the GRUForecast model.

        Args:
            input_size (int): Number of features in the input time series (default 1)
            hidden_size (int): Number of features in the GRU hidden state
            num_layers (int): Number of stacked GRU layers
            dropout (float): Dropout probability for inter-layer connection (if num_layers>1)
            horizon (int): Number of future steps to predict (multi-step). horizon=1 for one-step.
        """
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Final linear layer maps hidden state to horizon outputs
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        """
        Forward pass through the GRU and linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
        Returns:
            torch.Tensor: Forecast tensor of shape (batch_size, horizon)
        """
        # Pass through GRU
        out, _ = self.gru(x)               # out shape: (batch, seq_len, hidden_size)
        last = out[:, -1, :]               # take the last time step: (batch, hidden_size)
        y = self.fc(last)                  # project to horizon: (batch, horizon)
        return y


class GRUSeq2Seq(nn.Module):
    """
    Sequence-to-sequence GRU model for multi-step forecasting using a decoder loop.
    """
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        horizon: int = 1
    ):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.decoder_cell = nn.GRUCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, target=None, teacher_forcing_prob: float = 0.0):
        """
        Forward pass with optional teacher forcing for Seq2Seq.

        Args:
            x (torch.Tensor): Input tensor (batch, seq_len, input_size)
            target (torch.Tensor, optional): Ground truth future values (batch, horizon)
            teacher_forcing_prob (float): Probability of using true target as next input
        Returns:
            torch.Tensor: Forecast tensor (batch, horizon)
        """
        batch_size = x.size(0)
        # Encode input
        _, hidden = self.encoder(x)      # hidden: (num_layers, batch, hidden_size)
        # Initialize decoder input as last true input step
        decoder_input = x[:, -1, :]      # (batch, input_size)
        outputs = []
        # Flatten hidden for GRUCell if multiple layers
        h = hidden[-1]                   # use top layer state: (batch, hidden_size)
        for t in range(self.horizon):
            h = self.decoder_cell(decoder_input, h)
            out_t = self.fc(h)           # (batch, 1)
            outputs.append(out_t)
            # Decide next input
            if target is not None and torch.rand(1).item() < teacher_forcing_prob:
                decoder_input = target[:, t].unsqueeze(1)
            else:
                decoder_input = out_t
        # Concatenate outputs: list of (batch,1) → (batch,horizon)
        return torch.cat(outputs, dim=1)
