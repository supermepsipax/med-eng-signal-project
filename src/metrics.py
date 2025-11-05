"""
Advanced Metrics Module for Sleep Stage Classification

Core metric calculations without print statements.
Visualization and reporting handled by respective modules.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings


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


def plot_roc_curves(y_true, y_pred_proba, n_classes=5, stage_names=None):
    """
    Plot ROC curves for each class.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        n_classes (int): Number of classes
        stage_names (list): Names of sleep stages
    
    Returns:
        matplotlib.figure.Figure: ROC curve figure
    """
    if stage_names is None:
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                   label=f'{stage_names[i]} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves - Multi-Class Sleep Stage Classification', 
                fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


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


def plot_feature_importance(feature_importance_dict, top_n=20):
    """
    Plot top N most important features.
    
    Args:
        feature_importance_dict (dict): Output from calculate_feature_importance()
        top_n (int): Number of top features to display
    
    Returns:
        matplotlib.figure.Figure or None: Feature importance plot (None if not available)
    """
    if not feature_importance_dict['available']:
        return None
    
    importances = feature_importance_dict['importances']
    sorted_indices = feature_importance_dict['sorted_indices']
    feature_names = feature_importance_dict['feature_names']
    
    # Limit to top N features
    top_indices = sorted_indices[:top_n]
    top_importances = importances[top_indices]
    
    # Create feature labels
    if feature_names is not None:
        top_labels = [feature_names[i] for i in top_indices]
    else:
        top_labels = [f'Feature {i}' for i in top_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    y_pos = np.arange(len(top_labels))
    
    ax.barh(y_pos, top_importances, color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    return fig


def generate_iteration1_feature_names(n_channels=2):
    """
    Generate feature names for Iteration 1 (time-domain features).
    
    Args:
        n_channels (int): Number of EEG channels (default: 2)
    
    Returns:
        list: List of feature names
    """
    base_features = [
        'mean', 'median', 'std', 'variance', 'rms', 'min', 'max', 'range',
        'skewness', 'kurtosis', 'zero_crossings', 'hjorth_activity',
        'hjorth_mobility', 'hjorth_complexity', 'total_energy', 'mean_power'
    ]
    
    feature_names = []
    for ch in range(n_channels):
        for feat in base_features:
            feature_names.append(f'EEG_ch{ch+1}_{feat}')
    
    return feature_names


def get_kappa_interpretation(kappa):
    """
    Get interpretation string for Cohen's Kappa value.
    
    Args:
        kappa (float): Cohen's Kappa coefficient
    
    Returns:
        str: Interpretation string
    """
    if kappa < 0.00:
        return "Poor agreement"
    elif kappa <= 0.20:
        return "Slight agreement"
    elif kappa <= 0.40:
        return "Fair agreement"
    elif kappa <= 0.60:
        return "Moderate agreement"
    elif kappa <= 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"