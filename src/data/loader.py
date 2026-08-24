import os
import hashlib
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from src.config import config
from src.data.preprocessor import TextPreprocessor

def load_raw_dataset(csv_path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads and cleans invalid rows from the WELFake dataset.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # 1. Standardize columns
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # 2. Impute nulls
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    
    # 3. Filter out zero-signal rows (both title and text empty)
    mask = (df['title'].str.strip() != '') | (df['text'].str.strip() != '')
    df = df[mask].reset_index(drop=True)
    
    # 4. Ensure integer labels
    df['label'] = df['label'].astype(int)
    return df

def get_stratified_splits(
    df: pd.DataFrame, 
    test_size: float = 0.15, 
    val_size: float = 0.15, 
    random_state: int = config.RANDOM_SEED
) -> Dict[str, pd.DataFrame]:
    """
    Generates leak-free, stratified Train, Validation, and Test splits.
    """
    # First split off Test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
    
    # Calculate relative validation size
    val_relative_size = val_size / (1.0 - test_size)
    
    # Split Train and Validation sets
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df['label'],
        random_state=random_state
    )
    
    return {
        'train': train_df.reset_index(drop=True),
        'val': val_df.reset_index(drop=True),
        'test': test_df.reset_index(drop=True)
    }

def prepare_split_data(title_repeat: int = 2) -> Dict[str, Tuple[list, np.ndarray]]:
    """
    Loads dataset, applies preprocessor, and returns (X_texts, y_labels) for train, val, test.
    """
    df = load_raw_dataset()
    splits = get_stratified_splits(df)
    preprocessor = TextPreprocessor(title_repeat=title_repeat)
    
    processed_splits = {}
    for split_name, split_df in splits.items():
        X = preprocessor.transform(split_df)
        y = split_df['label'].values
        processed_splits[split_name] = (X, y)
        
    return processed_splits
