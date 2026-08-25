import sys
import joblib
import pandas as pd
from src.data.loader import prepare_split_data
from src.evaluation.evaluator import evaluate_predictions

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

splits = prepare_split_data(title_repeat=1)
X_test, y_test = splits['test']

pipe = joblib.load("artifacts/best_model.joblib")
test_vec = pipe.vectorizer.transform(X_test)
preds = pipe.clf.predict(test_vec)
proba = pipe.clf.predict_proba(test_vec)[:, 1]

metrics = evaluate_predictions(y_test, preds, proba, model_name="Production Pipeline")
cm = metrics['confusion_matrix']
tn, fp = cm[0]
fn, tp = cm[1]

print(f"   True Negatives (TN): {tn:,} / 5,255 ({tn/5255*100:.2f}% Real correctly classified)")
print(f"   True Positives (TP): {tp:,} / 5,566 ({tp/5566*100:.2f}% Fake correctly classified)")
print(f"   False Positives(FP): {fp:,} ({fp/5255*100:.2f}% Real misclassified as Fake)")
print(f"   False Negatives(FN): {fn:,} ({fn/5566*100:.2f}% Fake misclassified as Real)")
