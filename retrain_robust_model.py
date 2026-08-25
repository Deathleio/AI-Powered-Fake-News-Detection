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
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.config import config
from src.data.loader import prepare_split_data
from src.data.preprocessor import fuse_title_body
from src.models.baselines import FakeNewsPipeline
from src.evaluation.evaluator import evaluate_predictions, print_metrics_summary

def train_robust_model():
    print("================================================================", flush=True)
    print("[*] TRAINING MODEL WITH STRICT 0=FAKE AND 1=REAL GROUND TRUTH", flush=True)
    print("================================================================", flush=True)

    # 1. Load Data (0 = Fake, 1 = Real)
    print("Loading data splits...", flush=True)
    splits = prepare_split_data(title_repeat=1)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    # 2. Fit TF-IDF Vectorizer
    print("Fitting TF-IDF vectorizer (stop_words='english', ngram_range=(1, 2))...", flush=True)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
        min_df=3,
        stop_words='english',
        lowercase=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # 3. Train Calibrated Passive-Aggressive
    print("Training Calibrated Passive-Aggressive...", flush=True)
    base_pa = PassiveAggressiveClassifier(C=0.3, max_iter=1000, random_state=42)
    clf_pa = CalibratedClassifierCV(estimator=base_pa, method='sigmoid', cv=3)
    clf_pa.fit(X_train_vec, y_train)

    # 4. Train Regularized Logistic Regression
    print("Training Regularized Logistic Regression...", flush=True)
    clf_lr = LogisticRegression(C=2.0, solver='lbfgs', max_iter=500, random_state=42)
    clf_lr.fit(X_train_vec, y_train)

    # 5. Evaluate on 10,821 holdout test set
    pa_preds = clf_pa.predict(X_test_vec)
    pa_proba = clf_pa.predict_proba(X_test_vec)[:, 1]
    pa_metrics = evaluate_predictions(y_test, pa_preds, pa_proba, model_name="Calibrated Passive-Aggressive (0=Fake, 1=Real)")
    print_metrics_summary(pa_metrics)

    # 6. Save best pipeline
    best_pipe = FakeNewsPipeline(vectorizer, clf_pa)
    best_pipe.save("artifacts/best_model.joblib")
    best_pipe.save("artifacts/model_passive_aggressive.joblib")
    best_pipe.save("artifacts/model_logistic_regression.joblib")

    # 7. Test Bobby Jindal directly!
    print("\n================================================================", flush=True)
    print("[*] TESTING BOBBY JINDAL AND SAMPLES (0=FAKE, 1=REAL)", flush=True)
    print("================================================================", flush=True)

    test_cases = [
        ("Bobby Jindal (Row 3)", "Bobby Jindal, raised Hindu, uses story of Christian conversion to woo evangelicals for potential 2016 bid", "A dozen politically active pastors came here for a private dinner Friday night to hear a conversion story unique in the context of presidential politics: how Louisiana Gov. Bobby Jindal traveled from Hinduism to Protestant Christianity and, ultimately, became what he calls an evangelical Catholic. Over two hours, Jindal, 42, recalled talking with a girl in high school who wanted to save my soul, reading the Bible in a closet so his parents would not see him and feeling a stir while watching a movie during his senior year that depicted Jesus on the cross. He told the pastors the story of that night, and how it led him on a path that he says now drives his policy views, from abortion to religious liberty.", False),
        ("Clickbait Conspiracy", "SHOCKING BOMBSHELL: Secret Globalist Plot Leaked To Ban All Cash And Confiscate Savings By Next Week [VIDEO]", "UNBELIEVABLE! Top secret government whistleblowers have exposed an explosive classified document proving corrupt globalist elites are orchestrating a total financial blackout to seize your private bank accounts! Mainstream media refuses to report this terrifying scheme. Watch the emergency video before censors take it down!", False),
    ]

    for name, title, text, expected_real in test_cases:
        fused = fuse_title_body(title, text, title_repeat=1)
        proba_real = float(best_pipe.predict_proba([fused])[0, 1])
        proba_fake = float(best_pipe.predict_proba([fused])[0, 0])
        is_fake = proba_fake >= 0.5
        conf = proba_fake if is_fake else proba_real
        verdict = "Fake News" if is_fake else "Real News"
        print(f"[{name}] -> Predicted: {verdict} ({conf*100:.2f}% Conf) | Fake Proba: {proba_fake:.4f}, Real Proba: {proba_real:.4f}", flush=True)

if __name__ == '__main__':
    train_robust_model()
