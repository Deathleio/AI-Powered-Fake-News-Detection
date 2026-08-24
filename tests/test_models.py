import unittest
import numpy as np
import torch
from src.models.baselines import FakeNewsPipeline, build_vectorizer
from src.models.lstm_attention import TextVocabulary, BiLSTMAttentionClassifier
from src.models.cnn_bilstm import CNNBiLSTMClassifier
from src.evaluation.evaluator import evaluate_predictions

class TestModels(unittest.TestCase):
    def test_evaluator_metrics(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.4])
        metrics = evaluate_predictions(y_true, y_pred, y_prob, model_name="TestModel")
        self.assertEqual(metrics['accuracy'], 0.8)
        self.assertIn('macro_f1', metrics)
        self.assertIn('roc_auc', metrics)

    def test_vocabulary_and_lstm(self):
        texts = ["breaking news headline", "regular factual report", "another sample text"]
        vocab = TextVocabulary(max_vocab_size=100)
        vocab.build_vocab(texts)
        self.assertGreater(len(vocab.word2idx), 2)
        
        encoded = vocab.encode("breaking news", max_len=10)
        self.assertEqual(len(encoded), 10)
        
        model = BiLSTMAttentionClassifier(vocab_size=len(vocab.word2idx), embedding_dim=16, hidden_dim=16)
        input_tensor = torch.tensor([encoded], dtype=torch.long)
        logits = model(input_tensor)
        self.assertEqual(logits.shape, (1,))

    def test_cnn_bilstm_forward(self):
        model = CNNBiLSTMClassifier(vocab_size=50, embedding_dim=16, num_filters=8, filter_sizes=(3, 5), lstm_hidden=16)
        input_tensor = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        logits = model(input_tensor)
        self.assertEqual(logits.shape, (1,))

if __name__ == '__main__':
    unittest.main()
