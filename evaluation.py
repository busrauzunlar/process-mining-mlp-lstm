import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

def evaluate_model(model, X_test, y_test, name="Model"):
    """
    Evaluates model performance and returns a dictionary of metrics.
    Handles ROC-AUC safety and confusion matrix alignment.
    """
    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    # Force 2x2 confusion matrix using labels=[0,1]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    specificity = tn / (tn + fp + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    # Safe ROC-AUC calculation
    try:
        roc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc = 0.5 # Default or neutral value when only one class is present

    print(f"\n===== {name} =====")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc:.4f}")

    return {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc,
        "y_prob": y_prob,
        "y_pred": y_pred
    }

def plot_cm(y_test, y_pred, title, filename):
    """
    Plots and saves confusion matrix heatmap.
    """
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", filename))
    plt.close()

def plot_roc_curves(results_list, title="ROC Curves", filename="roc_curves.png"):
    """
    Plots ROC curves for multiple models on a single graph and saves it.
    """
    plt.figure(figsize=(8, 6))
    for res in results_list:
        name = res["name"]
        y_prob = res["y_prob"]
        y_true = res["y_true"]
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
        except ValueError:
            # If only one class is present in y_true, skip or plot diagonal
            pass
            
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", filename))
    plt.close()

def plot_pr_curves(results_list, title="Precision-Recall Curves", filename="pr_curves.png"):
    """
    Plots Precision-Recall curves for multiple models on a single graph and saves it.
    """
    plt.figure(figsize=(8, 6))
    for res in results_list:
        name = res["name"]
        y_prob = res["y_prob"]
        y_true = res["y_true"]
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", filename))
    plt.close()

def plot_training_history(history, model_name, filename_prefix):
    """
    Plots and saves loss and accuracy history for model training.
    """
    # Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="Train Acc")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title(f"{model_name} Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", f"{filename_prefix}_accuracy.png"))
    plt.close()

    # Loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"{model_name} Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", f"{filename_prefix}_loss.png"))
    plt.close()

def compare_models(results_list, title="Model Comparison", filename="model_comparison.png"):
    """
    Compares MLP, LSTM, and GRU models across multiple metrics using a bar chart and saves it.
    """
    metrics = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    
    n_models = len(results_list)
    x = np.arange(len(metrics))
    width = 0.8 / n_models
    
    plt.figure(figsize=(12, 6))
    
    for i, res in enumerate(results_list):
        vals = [res[m] for m in metrics]
        offset = (i - (n_models - 1) / 2) * width
        plt.bar(x + offset, vals, width=width, label=res["name"])
        
    plt.xticks(x, [m.upper() for m in metrics], rotation=15)
    plt.ylim([0.0, 1.1])
    plt.ylabel("Score")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", filename))
    plt.close()