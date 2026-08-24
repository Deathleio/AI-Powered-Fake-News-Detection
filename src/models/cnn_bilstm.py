import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.config import config

class CNNBiLSTMClassifier(nn.Module):
    """
    Multi-Scale 1D-CNN + BiLSTM Hybrid Deep Learning Architecture.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = config.EMBEDDING_DIM,
        num_filters: int = config.CNN_FILTERS,
        filter_sizes: tuple = config.CNN_KERNEL_SIZES,
        lstm_hidden: int = config.LSTM_HIDDEN_DIM,
        dropout: float = config.DROPOUT
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Parallel 1D Convolutions with padding='same' for consistent temporal dimension
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k, padding='same')
            for k in filter_sizes
        ])
        
        conv_out_dim = num_filters * len(filter_sizes)
        
        self.lstm = nn.LSTM(
            input_size=conv_out_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch_size, seq_len)
        embeds = self.embedding(input_ids).transpose(1, 2) # (batch_size, embed_dim, seq_len)
        
        # Multi-scale conv activations
        conv_outs = [F.relu(conv(embeds)) for conv in self.convs]
        combined = torch.cat(conv_outs, dim=1).transpose(1, 2) # (batch_size, seq_len, total_filters)
        
        lstm_out, _ = self.lstm(combined) # (batch_size, seq_len, lstm_hidden * 2)
        
        pooled, _ = torch.max(lstm_out, dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        return logits
