import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import pandas as pd

def plot_data_quality(df, save_path='data_quality.png'):
    """Checks for missing values """
    plt.figure(figsize=(10, 6))
    missing_values = df.isnull().sum()
    missing_values.plot(kind='bar', color='salmon')
    plt.title('Data Quality Analysis: Missing Values per Feature', fontsize=14)
    plt.xlabel('Medical Features', fontsize=12)
    plt.ylabel('Count of Missing Values', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_performance_plots(model, X_test, y_test, feature_names):

    # 1. Feature Importance
    plt.figure(figsize=(10, 8))
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    importances.plot(kind='barh', color='teal')
    plt.title('Feature Importance: Key Drivers of Heart Disease Model', fontsize=14)
    plt.xlabel('Importance Score (Information Gain)', fontsize=12)
    plt.ylabel('Medical Attributes', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    # 2. Confusion Matrix
    plt.figure(figsize=(8, 6))
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Model Prediction Error Distribution', fontsize=14)
    plt.xlabel('Predicted Condition (0: Healthy, 1: At Risk)', fontsize=12)
    plt.ylabel('Actual Condition (0: Healthy, 1: At Risk)', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_roc_curve(model, X_test, y_test, save_path='roc_curve.png'):
    """Plots the ROC curve ."""
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Area = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve: Diagnostic Separation Power', fontsize=14)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall(model, X_test, y_test, save_path='precision_recall.png'):
    """Plots the Precision-Recall """
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2)
    plt.title('Precision-Recall Curve: Performance on Minority Class (Sick)', fontsize=14)
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_correlation_heatmap(df, save_path='correlation_heatmap.png'):
    """Visualizes the correlation heatmap"""
    plt.figure(figsize=(14, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix: Linear Relationships Between Medical Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()