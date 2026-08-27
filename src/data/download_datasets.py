import os
import sys
import urllib.request
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATASET_DIR = os.path.abspath("dataset_study")
RAW_DIR = os.path.join(DATASET_DIR, "raw")
OUTPUT_PATH = os.path.join(DATASET_DIR, "unified_multidomain_dataset.csv")

os.makedirs(RAW_DIR, exist_ok=True)

LIAR_URLS = {
    "train": "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv",
    "test": "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv",
    "valid": "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv"
}

COAID_URLS = {
    "real": "https://raw.githubusercontent.com/cuilimeng/CoAID/master/05-01-2020/NewsRealCOVID-19.csv",
    "fake": "https://raw.githubusercontent.com/cuilimeng/CoAID/master/05-01-2020/NewsFakeCOVID-19.csv"
}

def download_file(url: str, dest_path: str):
    if not os.path.exists(dest_path):
        print(f"Downloading {os.path.basename(dest_path)} from {url}...", flush=True)
        urllib.request.urlretrieve(url, dest_path)
    else:
        print(f"File already exists: {os.path.basename(dest_path)}", flush=True)

def load_and_process_liar() -> pd.DataFrame:
    """
    Downloads and standardizes the LIAR dataset (12.8k short political claims).
    Labels:
      - true, mostly-true -> 1 (Real)
      - false, pants-fire, barely-true -> 0 (Fake)
      - half-true -> omitted for clear binary margin
    """
    print("\nProcessing LIAR Dataset (Short Claims & PolitiFact Statements)...", flush=True)
    frames = []
    cols = [
        "id", "label", "statement", "subject", "speaker", "job_title", 
        "state_info", "party", "barely_true_cnt", "false_cnt", 
        "half_true_cnt", "mostly_true_cnt", "pants_on_fire_cnt", "context"
    ]
    
    for split, url in LIAR_URLS.items():
        dest = os.path.join(RAW_DIR, f"liar_{split}.tsv")
        download_file(url, dest)
        df_split = pd.read_csv(dest, sep='\t', header=None, names=cols, on_bad_lines='skip')
        frames.append(df_split)
        
    df_liar = pd.concat(frames, ignore_index=True)
    
    # Binary mapping
    real_labels = {"true", "mostly-true"}
    fake_labels = {"false", "pants-fire", "barely-true"}
    
    df_liar = df_liar[df_liar['label'].isin(real_labels | fake_labels)].copy()
    df_liar['standard_label'] = df_liar['label'].apply(lambda x: 1 if x in real_labels else 0)
    
    df_clean = pd.DataFrame({
        'title': df_liar['statement'].fillna(''),
        'text': (df_liar['speaker'].fillna('') + " in " + df_liar['context'].fillna('')).replace(' in ', ''),
        'label': df_liar['standard_label'],
        'domain': 'politics_short_claim',
        'source': 'LIAR_PolitiFact'
    })
    print(f"Processed LIAR: {len(df_clean)} samples ({int((df_clean['label']==1).sum())} Real, {int((df_clean['label']==0).sum())} Fake)", flush=True)
    return df_clean

def load_and_process_coaid() -> pd.DataFrame:
    """
    Downloads and standardizes the CoAID Healthcare / COVID Misinformation Dataset.
    """
    print("\nProcessing CoAID Dataset (Health & Medical Misinformation)...", flush=True)
    real_dest = os.path.join(RAW_DIR, "coaid_real.csv")
    fake_dest = os.path.join(RAW_DIR, "coaid_fake.csv")
    
    download_file(COAID_URLS["real"], real_dest)
    download_file(COAID_URLS["fake"], fake_dest)
    
    df_real = pd.read_csv(real_dest)
    df_fake = pd.read_csv(fake_dest)
    
    records = []
    for _, row in df_real.iterrows():
        title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
        text = str(row.get('content', '')) if pd.notna(row.get('content')) else ''
        if title or text:
            records.append({'title': title, 'text': text, 'label': 1, 'domain': 'healthcare_science', 'source': 'CoAID_Real'})
            
    for _, row in df_fake.iterrows():
        title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
        text = str(row.get('content', '')) if pd.notna(row.get('content')) else ''
        if title or text:
            records.append({'title': title, 'text': text, 'label': 0, 'domain': 'healthcare_science', 'source': 'CoAID_Fake'})
            
    df_coaid = pd.DataFrame(records)
    print(f"Processed CoAID: {len(df_coaid)} samples ({int((df_coaid['label']==1).sum())} Real, {int((df_coaid['label']==0).sum())} Fake)", flush=True)
    return df_coaid

def load_and_process_welfake() -> pd.DataFrame:
    """
    Standardizes the WELFake baseline dataset (72,134 long articles).
    In WELFake raw: 0 = Real, 1 = Fake. Standardized: 1 = Real, 0 = Fake.
    """
    print("\nProcessing WELFake Dataset (General Long-Form News)...", flush=True)
    welfake_path = "WELFake_Dataset.csv"
    if not os.path.exists(welfake_path):
        print(f"Warning: {welfake_path} not found in root.", flush=True)
        return pd.DataFrame()
        
    df = pd.read_csv(welfake_path)
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    mask = (df['title'].str.strip() != '') | (df['text'].str.strip() != '')
    df = df[mask].copy()
    
    # 0 = Fake, 1 = Real
    df['standard_label'] = 1 - df['label'].astype(int)
    
    df_clean = pd.DataFrame({
        'title': df['title'],
        'text': df['text'],
        'label': df['standard_label'],
        'domain': 'general_longform_news',
        'source': 'WELFake'
    })
    print(f"Processed WELFake: {len(df_clean)} samples ({int((df_clean['label']==1).sum())} Real, {int((df_clean['label']==0).sum())} Fake)", flush=True)
    return df_clean

def build_unified_dataset():
    print("================================================================", flush=True)
    print("[*] BUILDING UNIFIED MULTI-DOMAIN FAKE NEWS DATASET", flush=True)
    print("================================================================", flush=True)
    
    df_liar = load_and_process_liar()
    df_coaid = load_and_process_coaid()
    df_welfake = load_and_process_welfake()
    
    dfs = [d for d in [df_welfake, df_liar, df_coaid] if not d.empty]
    df_unified = pd.concat(dfs, ignore_index=True)
    
    # Drop duplicates
    df_unified = df_unified.drop_duplicates(subset=['title', 'text']).reset_index(drop=True)
    
    # Save output
    df_unified.to_csv(OUTPUT_PATH, index=False)
    
    total = len(df_unified)
    real_cnt = int((df_unified['label'] == 1).sum())
    fake_cnt = int((df_unified['label'] == 0).sum())
    
    print("\n================================================================", flush=True)
    print(f"[*] UNIFIED DATASET CREATED: {OUTPUT_PATH}", flush=True)
    print(f"  Total Diverse Samples: {total:,}")
    print(f"  Real News (Class 1):   {real_cnt:,} ({real_cnt/total*100:.2f}%)")
    print(f"  Fake News (Class 0):   {fake_cnt:,} ({fake_cnt/total*100:.2f}%)")
    print(f"  Domain Breakdown:")
    for dom, cnt in df_unified['domain'].value_counts().items():
        print(f"    - {dom}: {cnt:,} samples ({cnt/total*100:.1f}%)")
    print("================================================================\n", flush=True)

if __name__ == '__main__':
    build_unified_dataset()
