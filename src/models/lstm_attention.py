import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import numpy as np
from src.config import config

class TextVocabulary:
    """
    Fast, lightweight word-to-index vocabulary builder.
    """
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, max_vocab_size: int = 40000):
        self.max_vocab_size = max_vocab_size
        self.word2idx: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word: Dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def build_vocab(self, texts: List[str]):
        word_counts = {}
        for text in texts:
            tokens = re.findall(r'\w+', text.lower())
            for t in tokens:
                word_counts[t] = word_counts.get(t, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.max_vocab_size - 2]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text: str, max_len: int = 300) -> List[int]:
        tokens = re.findall(r'\w+', text.lower())
        encoded = [self.word2idx.get(t, 1) for t in tokens[:max_len]]
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))
        return encoded

    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.word2idx, f)

    @classmethod
    def load(cls, filepath: str) -> "TextVocabulary":
        vocab = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab.word2idx = json.load(f)
        vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
        return vocab


class NewsTorchDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[np.ndarray], vocab: TextVocabulary, max_len: int = 300):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = self.vocab.encode(self.texts[idx], max_len=self.max_len)
        item = {'input_ids': torch.tensor(seq, dtype=torch.long)}
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class BahdanauAttention(nn.Module):
    """
    Bahdanau Additive Attention mechanism for RNN hidden states.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # rnn_outputs: (batch_size, seq_len, hidden_dim)
        scores = self.v(torch.tanh(self.W(rnn_outputs))) # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1) # (batch_size, seq_len, 1)
        context = torch.sum(weights * rnn_outputs, dim=1) # (batch_size, hidden_dim)
        return context, weights


class BiLSTMAttentionClassifier(nn.Module):
    """
    Bidirectional LSTM with Bahdanau Attention and Multi-Layer Classification Head.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = config.EMBEDDING_DIM,
        hidden_dim: int = config.LSTM_HIDDEN_DIM,
        num_layers: int = 2,
        dropout: float = config.DROPOUT
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout2d(0.2)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.attention = BahdanauAttention(hidden_dim * 2)
        
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch_size, seq_len)
        embeds = self.embedding(input_ids) # (batch_size, seq_len, embed_dim)
        
        lstm_out, _ = self.lstm(embeds) # (batch_size, seq_len, hidden_dim * 2)
        context, _ = self.attention(lstm_out) # (batch_size, hidden_dim * 2)
        
        logits = self.fc_head(context).squeeze(-1) # (batch_size,)
        return logits
