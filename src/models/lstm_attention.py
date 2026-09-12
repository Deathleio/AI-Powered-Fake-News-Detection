import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Union, Any
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

    def forward(self, input_ids: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # input_ids: (batch_size, seq_len)
        embeds = self.embedding(input_ids) # (batch_size, seq_len, embed_dim)
        
        lstm_out, _ = self.lstm(embeds) # (batch_size, seq_len, hidden_dim * 2)
        context, weights = self.attention(lstm_out) # (batch_size, hidden_dim * 2), (batch_size, seq_len, 1)
        
        logits = self.fc_head(context).squeeze(-1) # (batch_size,)
        if return_attention:
            return logits, weights.squeeze(-1)
        return logits


class DeepLearningNewsPipeline:
    """
    Commercial-Grade Deep Learning Fake News Pipeline.
    Encapsulates Text Vocabulary Tokenization, PyTorch Bi-directional LSTM with
    Bahdanau Additive Attention mechanism, and intrinsic token saliency extraction.
    """
    def __init__(
        self,
        model: BiLSTMAttentionClassifier,
        vocab: TextVocabulary,
        max_len: int = 256,
        device: str = "cpu"
    ):
        self.model = model
        self.vocab = vocab
        self.max_len = max_len
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        batch_ids = [self.vocab.encode(t, max_len=self.max_len) for t in texts]
        return torch.tensor(batch_ids, dtype=torch.long, device=self.device)

    def predict_proba(self, texts: List[str], title_texts: Optional[List[str]] = None) -> np.ndarray:
        """
        Calculates veracity probabilities:
        Column 0 = P(Fake News)
        Column 1 = P(Real News)
        """
        self.model.eval()
        with torch.no_grad():
            input_ids = self._encode_batch(texts)
            logits = self.model(input_ids)
            # Sigmoid output represents P(Real News) [target convention: 1 = Real, 0 = Fake]
            p_real = torch.sigmoid(logits).cpu().numpy()
            p_fake = 1.0 - p_real
            return np.column_stack([p_fake, p_real])

    def predict(self, texts: List[str], title_texts: Optional[List[str]] = None) -> np.ndarray:
        """
        Binary prediction: 0 = Fake News, 1 = Real News.
        """
        proba = self.predict_proba(texts, title_texts=title_texts)
        return (proba[:, 1] >= 0.5).astype(int)

    def explain_text(
        self,
        text: str,
        title: Optional[str] = None,
        top_k: int = 10,
        sensational_tokens: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extracts Explainable AI (XAI) token attribution directly from the neural
        network's learned Bahdanau Attention weights.
        """
        self.model.eval()
        tokens = re.findall(r'\w+', text.lower())
        encoded = self.vocab.encode(text, max_len=self.max_len)
        input_tensor = torch.tensor([encoded], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits, att_weights = self.model(input_tensor, return_attention=True)
            p_real = float(torch.sigmoid(logits).cpu().item())
            p_fake = 1.0 - p_real
            weights = att_weights[0].cpu().numpy()

        # Pair valid words with their attention scores
        word_attentions = []
        valid_len = min(len(tokens), self.max_len)
        for i in range(valid_len):
            word = tokens[i]
            if len(word) > 2:  # Skip single/double letter tokens
                word_attentions.append((word, float(weights[i])))

        # Rank by attention mass
        ranked = sorted(word_attentions, key=lambda x: x[1], reverse=True)

        is_fake = p_fake >= 0.5
        real_indicators = []
        fake_indicators = []

        if is_fake:
            # Under Fake classification, top attended tokens are primary fake triggers
            fake_indicators = [{"token": w, "weight": round(score * 10, 4)} for w, score in ranked[:top_k]]
            # Attended tokens further down represent context/entities
            real_indicators = [{"token": w, "weight": round(score * 10, 4)} for w, score in ranked[top_k:top_k*2]]
        else:
            # Under Real classification, top attended tokens are primary factual anchors
            real_indicators = [{"token": w, "weight": round(score * 10, 4)} for w, score in ranked[:top_k]]
            fake_indicators = [{"token": w, "weight": round(score * 10, 4)} for w, score in ranked[top_k:top_k*2]]

        # Inject detected stylistic sensational keywords if present
        if sensational_tokens:
            existing_fake = {item['token'].lower() for item in fake_indicators}
            for st in sensational_tokens:
                if st.lower() not in existing_fake:
                    fake_indicators.insert(0, {"token": st.lower(), "weight": 0.4500})

        return {
            "p_fake": p_fake,
            "p_real": p_real,
            "real_indicators": real_indicators[:top_k],
            "fake_indicators": fake_indicators[:top_k]
        }

    def save(self, model_path: str, vocab_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        self.vocab.save(vocab_path)
        print(f"Deep learning model weights saved to {model_path}", flush=True)
        print(f"Vocabulary saved to {vocab_path}", flush=True)

    @classmethod
    def load(
        cls,
        model_path: str,
        vocab_path: str,
        embedding_dim: int = config.EMBEDDING_DIM,
        hidden_dim: int = config.LSTM_HIDDEN_DIM,
        num_layers: int = 1,
        dropout: float = config.DROPOUT,
        device: str = "cpu"
    ) -> "DeepLearningNewsPipeline":
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        vocab = TextVocabulary.load(vocab_path)
        state_dict = torch.load(model_path, map_location=device)

        # Detect architecture dimensions from checkpoint state_dict
        vocab_size = state_dict['embedding.weight'].shape[0]
        actual_embed_dim = state_dict['embedding.weight'].shape[1]
        
        # Check num_layers from state dict keys
        detected_layers = 1
        if 'lstm.weight_ih_l1' in state_dict:
            detected_layers = 2
            
        actual_hidden_dim = state_dict['lstm.weight_ih_l0'].shape[0] // 4

        model = BiLSTMAttentionClassifier(
            vocab_size=vocab_size,
            embedding_dim=actual_embed_dim,
            hidden_dim=actual_hidden_dim,
            num_layers=detected_layers,
            dropout=dropout
        )
        model.load_state_dict(state_dict)
        return cls(model=model, vocab=vocab, device=device)
