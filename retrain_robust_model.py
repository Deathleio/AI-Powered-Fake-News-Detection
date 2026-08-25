import sys
import time
import json
import joblib
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.config import config
from src.data.loader import prepare_split_data
from src.data.preprocessor import fuse_title_body
from src.models.baselines import FakeNewsPipeline
from src.evaluation.evaluator import evaluate_predictions, print_metrics_summary

def train_robust_model():
    print("================================================================", flush=True)
    print("[*] TRAINING REGULARIZED, GENERALIZED FAKE NEWS MODEL", flush=True)
    print("================================================================", flush=True)

    # 1. Load Data (0 = Real News, 1 = Fake News in WELFake)
    print("Loading stratified dataset splits...", flush=True)
    splits = prepare_split_data(title_repeat=1)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    # 2. Fit Regularized TF-IDF Vectorizer
    # min_df=5 prevents overfitting to rare spurious tokens; max_df=0.90 filters corpus boilerplate
    print("Fitting Regularized TF-IDF (min_df=5, max_df=0.90, max_features=35000, sublinear_tf=True)...", flush=True)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=35000,
        sublinear_tf=True,
        min_df=5,
        max_df=0.90,
        stop_words='english',
        lowercase=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # 3. Train Regularized Logistic Regression (C=0.8 with L2 penalty for smooth generalization)
    print("Training Regularized Logistic Regression (C=0.8, penalty='l2')...", flush=True)
    clf_lr = LogisticRegression(C=0.8, penalty='l2', solver='lbfgs', max_iter=300, random_state=42)
    clf_lr.fit(X_train_vec, y_train)

    # 4. Train ElasticNet SGD with Early Stopping (prevents overtraining)
    print("Training ElasticNet SGD with Early Stopping (alpha=1e-4, early_stopping=True)...", flush=True)
    clf_sgd = SGDClassifier(
        loss='log_loss',
        penalty='elasticnet',
        alpha=1e-4,
        l1_ratio=0.15,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=3,
        random_state=42
    )
    clf_sgd.fit(X_train_vec, y_train)

    # 5. Evaluate on Holdout Test Set (10,821 articles)
    print("\n--- Model Evaluation (Holdout Test Set) ---", flush=True)
    lr_preds = clf_lr.predict(X_test_vec)
    lr_proba = clf_lr.predict_proba(X_test_vec)[:, 1]
    lr_metrics = evaluate_predictions(y_test, lr_preds, lr_proba, model_name="Regularized Logistic Regression (C=0.8)")
    print_metrics_summary(lr_metrics)

    # 6. Save Regularized Pipeline
    best_pipe = FakeNewsPipeline(vectorizer, clf_lr)
    best_pipe.save("artifacts/best_model.joblib")
    best_pipe.save("artifacts/model_logistic_regression.joblib")
    
    sgd_pipe = FakeNewsPipeline(vectorizer, clf_sgd)
    sgd_pipe.save("artifacts/model_sgd_log.joblib")

    # 7. Validation against Adversarial and Generalization Test Cases
    print("\n================================================================", flush=True)
    print("[*] VALIDATION ON ADVERSARIAL & REAL GENERALIZATION SAMPLES", flush=True)
    print("================================================================", flush=True)

    test_cases = [
        ("Aliens on Hot Wheels (Fake)", "breaking report : the world is invaded my aliens on hot wheels", "the hot wheels brand has collaborated with aliens to inavade the world and start another holocaust"),
        ("World on Fire Bushfire (Fake)", "THE WORLD IS ON FIRE", "AUSTRALIA BUSHFIRE HAS TAKEN THE LIFE OF TRUMP WHO WAS DANCING WITH NETANYAHU"),
        ("Bobby Jindal (Real)", "Bobby Jindal, raised Hindu, uses story of Christian conversion to woo evangelicals for potential 2016 bid", "A dozen politically active pastors came here for a private dinner Friday night to hear a conversion story unique in the context of presidential politics: how Louisiana Gov. Bobby Jindal traveled from Hinduism to Protestant Christianity and, ultimately, became what he calls an evangelical Catholic. Over two hours, Jindal, 42, recalled talking with a girl in high school who wanted to save my soul, reading the Bible in a closet so his parents would not see him and feeling a stir while watching a movie during his senior year that depicted Jesus on the cross. He told the pastors the story of that night, and how it led him on a path that he says now drives his policy views, from abortion to religious liberty."),
        ("Federal Reserve (Real)", "Federal Reserve Holds Benchmark Interest Rates Steady Amid Cooling Inflation Data", "The Federal Reserve announced on Wednesday that benchmark interest rates will remain unchanged following a two-day policy meeting, according to official statements and central bank releases.")
    ]

    for name, title, text in test_cases:
        fused = fuse_title_body(title, text, title_repeat=1)
        proba_real = float(best_pipe.predict_proba([fused])[0, 1])
        proba_fake = float(best_pipe.predict_proba([fused])[0, 0])
        is_fake = proba_fake >= 0.5
        conf = proba_fake if is_fake else proba_real
        verdict = "Fake News" if is_fake else "Real News"
        print(f"[{name}] -> {verdict} ({conf*100:.2f}% Conf) | Fake: {proba_fake:.4f}, Real: {proba_real:.4f}", flush=True)

if __name__ == '__main__':
    train_robust_model()

