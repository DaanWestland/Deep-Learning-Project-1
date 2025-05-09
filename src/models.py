import torch
import torch.nn as nn


# ─── GRUForecast ────────────────────────────────────────────────────────────────

class GRUForecast(nn.Module):
    """
    A GRU-based model for time series forecasting that directly predicts a multi-step horizon.
    
    Args:
        input_size (int): Number of features per time step
        hidden_size (int): Number of features in the hidden state
        num_layers (int): Number of recurrent layers
        dropout (float): Dropout probability
        horizon (int): Number of steps to forecast ahead
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
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for n, p in self.gru.named_parameters():
            if 'weight' in n:
                nn.init.xavier_uniform_(p)
            elif 'bias' in n:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, horizon)
        """
        _, h_n = self.gru(x)                 # h_n: (num_layers, batch, hidden_size)
        last = h_n[-1]                       # (batch, hidden_size)
        out = self.fc(last)                  # (batch, horizon)
        return out



# ─── GRUSeq2Seq ────────────────────────────────────────────────────────────────

class GRUSeq2Seq(nn.Module):
    """
    A sequence-to-sequence GRU model for time series forecasting.
    
    Args:
        encoder_input_size (int): Number of input features
        hidden_size (int): Number of features in the hidden state
        num_layers (int): Number of recurrent layers
        dropout (float): Dropout probability
        horizon (int): Number of steps to forecast ahead
        decoder_input_size (int): Number of input features for decoder
    """
    def __init__(
        self,
        encoder_input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        horizon: int = 1,
        decoder_input_size: int = 1
    ):
        super().__init__()
        self.horizon = horizon

        self.encoder = nn.GRU(
            encoder_input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.decoder_cell = nn.GRUCell(decoder_input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, decoder_input_size)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for n, p in self.encoder.named_parameters():
            if 'weight' in n:
                nn.init.xavier_uniform_(p)
            elif 'bias' in n:
                nn.init.zeros_(p)
        for n, p in self.decoder_cell.named_parameters():
            if 'weight' in n:
                nn.init.xavier_uniform_(p)
            elif 'bias' in n:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor = None,
        teacher_forcing_prob: float = 0.0
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, F)
            target (torch.Tensor, optional): Target tensor of shape (batch, horizon) for teacher forcing
            teacher_forcing_prob (float): Probability of using teacher forcing
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, horizon)
        """
        x = self.dropout(x)
        _, hidden = self.encoder(x)
        h = hidden[-1]   # (batch, hidden_size)

        decoder_input = x[:, -1, 0].unsqueeze(-1)  # (batch, 1)
        outputs = []
        
        for t in range(self.horizon):
            h = self.decoder_cell(decoder_input, h)
            h = self.dropout(h)
            out_t = self.fc(h)                     # (batch, 1)
            outputs.append(out_t)

            if (target is not None and torch.rand(1).item() < teacher_forcing_prob):
                decoder_input = target[:, t].unsqueeze(-1)
            else:
                decoder_input = out_t

        return torch.cat(outputs, dim=1).squeeze(-1)



# ─── PositionalEncoding ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Injects sinusoidal positional embeddings, then dropout.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # precompute a large enough P×D
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, P, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # re-compute on‐the-fly if longer
            pos = torch.arange(seq_len, device=x.device).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, x.size(2), 2, device=x.device).float()
                * (-np.log(10000.0) / x.size(2))
            )
            pe = torch.zeros(seq_len, x.size(2), device=x.device)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            pe = pe.unsqueeze(0)
        else:
            pe = self.pe[:, :seq_len]

        return self.dropout(x + pe)

