import os
from dataclasses import dataclass

@dataclass
class Config:
    # Project paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_PATH: str = os.path.join(BASE_DIR, "WELFake_Dataset.csv")
    ARTIFACTS_DIR: str = os.path.join(BASE_DIR, "artifacts")
    
    # Split configuration
    RANDOM_SEED: int = 42
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    
    # Text Preprocessing & Tokenization Limits
    MAX_TITLE_WORDS: int = 64
    MAX_BODY_WORDS: int = 448
    MAX_SEQ_LENGTH: int = 512
    TFIDF_MAX_FEATURES: int = 50000
    
    # Model Architectures
    EMBEDDING_DIM: int = 128
    LSTM_HIDDEN_DIM: int = 128
    CNN_FILTERS: int = 64
    CNN_KERNEL_SIZES: tuple = (3, 4, 5)
    DROPOUT: float = 0.3
    
    # Training Parameters
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    NUM_EPOCHS: int = 8
    EARLY_STOPPING_PATIENCE: int = 2

config = Config()
