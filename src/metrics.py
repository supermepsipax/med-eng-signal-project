"""
Advanced Metrics Module for Sleep Stage Classification

Core metric calculations without print statements.
Visualization and reporting handled by respective modules.
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def calculate_cohens_kappa(y_true, y_pred):
    """
    Calculate Cohen's Kappa coefficient.
    
    Interpretation:
        < 0.00: Poor agreement
        0.00 - 0.20: Slight agreement
        0.21 - 0.40: Fair agreement
        0.41 - 0.60: Moderate agreement
        0.61 - 0.80: Substantial agreement
        0.81 - 1.00: Almost perfect agreement
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
    
    Returns:
        float: Cohen's Kappa coefficient
    """
    return cohen_kappa_score(y_true, y_pred)


def calculate_multiclass_roc_auc(y_true, y_pred_proba, n_classes=5):
    """
    Calculate ROC-AUC for each class using One-vs-Rest approach.
    
    Args:
        y_true (np.ndarray): True labels (shape: n_samples)
        y_pred_proba (np.ndarray): Predicted probabilities (shape: n_samples, n_classes)
        n_classes (int): Number of classes (default: 5 for sleep stages)
    
    Returns:
        dict: Dictionary containing:
            - 'per_class': list of AUC scores for each class
            - 'macro_avg': macro-averaged AUC across all classes
            - 'weighted_avg': weighted-averaged AUC (by class support)
    """
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    auc_scores = []
    
    for i in range(n_classes):
        try:
            if np.sum(y_true_bin[:, i]) > 0:
                auc_score = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                auc_scores.append(auc_score)
            else:
                auc_scores.append(np.nan)
        except ValueError:
            auc_scores.append(np.nan)
    
    # Calculate macro and weighted averages (excluding NaN values)
    valid_aucs = [score for score in auc_scores if not np.isnan(score)]
    
    if len(valid_aucs) > 0:
        macro_avg = np.mean(valid_aucs)
        class_counts = np.bincount(y_true, minlength=n_classes)
        valid_weights = [class_counts[i] for i, score in enumerate(auc_scores) 
                        if not np.isnan(score)]
        weighted_avg = np.average(valid_aucs, weights=valid_weights)
    else:
        macro_avg = np.nan
        weighted_avg = np.nan
    
    return {
        'per_class': auc_scores,
        'macro_avg': macro_avg,
        'weighted_avg': weighted_avg
    }


def calculate_feature_importance(model, feature_names=None):
    """
    Calculate feature importance from the trained model.
    
    Note: Only available for tree-based models (Random Forest, Decision Trees).
    
    Args:
        model: Trained sklearn model
        feature_names (list): List of feature names (optional)
    
    Returns:
        dict: Dictionary containing:
            - 'available': bool, whether feature importance is available
            - 'importances': np.ndarray or None
            - 'feature_names': list or None
            - 'sorted_indices': np.ndarray or None (indices sorted by importance)
    """
    result = {
        'available': False,
        'importances': None,
        'feature_names': feature_names,
        'sorted_indices': None
    }
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        result['available'] = True
        result['importances'] = importances
        result['sorted_indices'] = np.argsort(importances)[::-1]
    
    return result


