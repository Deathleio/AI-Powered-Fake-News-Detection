import json
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes a comprehensive suite of binary classification metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    f1_binary = float(f1_score(y_true, y_pred, average='binary', zero_division=0))
    
    auc = None
    if y_proba is not None:
        try:
            if y_proba.ndim == 2:
                prob_pos = y_proba[:, 1]
            else:
                prob_pos = y_proba
            auc = float(roc_auc_score(y_true, prob_pos))
        except Exception:
            auc = None

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=["Real (0)", "Fake (1)"], output_dict=True)

    metrics = {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": f1_macro,
        "binary_f1": f1_binary,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": report
    }
    return metrics

def print_metrics_summary(metrics: Dict[str, Any]):
    """
    Prints a formatted evaluation dashboard to stdout.
    """
    print(f"\n=======================================================")
    print(f"       EVALUATION REPORT: {metrics['model_name']}")
    print(f"=======================================================")
    print(f" Accuracy:       {metrics['accuracy'] * 100:.2f}%")
    print(f" Macro F1-Score: {metrics['macro_f1']:.4f}")
    print(f" Binary F1:      {metrics['binary_f1']:.4f}")
    print(f" Precision:      {metrics['precision']:.4f}")
    print(f" Recall:         {metrics['recall']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f" ROC-AUC Score:  {metrics['roc_auc']:.4f}")
    print(f" Confusion Matrix (TN, FP / FN, TP):")
    cm = metrics['confusion_matrix']
    print(f"   [Real/0]  TN: {cm[0][0]:<6}  FP: {cm[0][1]:<6}")
    print(f"   [Fake/1]  FN: {cm[1][0]:<6}  TP: {cm[1][1]:<6}")
    print(f"=======================================================\n")
