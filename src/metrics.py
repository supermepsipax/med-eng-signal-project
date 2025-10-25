"""
Advanced Metrics Module for Sleep Stage Classification

This module implements advanced performance metrics including:
- Cohen's Kappa (inter-rater agreement)
- ROC-AUC for each class (one-vs-rest)
- Feature importance analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings


def calculate_cohens_kappa(y_true, y_pred):
    """
    Calculate Cohen's Kappa coefficient.
    
    Cohen's Kappa measures inter-rater agreement, accounting for agreement
    occurring by chance. It's particularly important for sleep scoring where
    class imbalance is natural.
    
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
    kappa = cohen_kappa_score(y_true, y_pred)
    return kappa


def interpret_kappa(kappa):
    """
    Provide interpretation of Cohen's Kappa value.
    
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


def calculate_multiclass_roc_auc(y_true, y_pred_proba, n_classes=5):
    """
    Calculate ROC-AUC for each class using One-vs-Rest approach.
    
    For multi-class problems, we compute ROC-AUC for each class separately
    by treating it as a binary classification problem (class vs. all others).
    
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
    # Binarize the labels for One-vs-Rest
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Calculate AUC for each class
    auc_scores = []
    
    for i in range(n_classes):
        try:
            # Check if this class exists in y_true
            if np.sum(y_true_bin[:, i]) > 0:
                auc_score = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                auc_scores.append(auc_score)
            else:
                # Class not present in test set
                auc_scores.append(np.nan)
                warnings.warn(f"Class {i} not present in test set, AUC set to NaN")
        except ValueError as e:
            # Handle cases where AUC cannot be computed
            auc_scores.append(np.nan)
            warnings.warn(f"Could not compute AUC for class {i}: {e}")
    
    # Calculate macro and weighted averages (excluding NaN values)
    valid_aucs = [score for score in auc_scores if not np.isnan(score)]
    
    if len(valid_aucs) > 0:
        macro_avg = np.mean(valid_aucs)
        
        # Weighted average by class support
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
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve for each class
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:  # Only plot if class exists
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
    
    Note: Feature importance is only available for tree-based models
    (Random Forest, Decision Trees). For k-NN (Iteration 1), this will
    return a message indicating it's not applicable.
    
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
    
    # Check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        result['available'] = True
        result['importances'] = importances
        
        # Sort features by importance (descending)
        sorted_indices = np.argsort(importances)[::-1]
        result['sorted_indices'] = sorted_indices
    
    return result


def plot_feature_importance(feature_importance_dict, top_n=20):
    """
    Plot top N most important features.
    
    Args:
        feature_importance_dict (dict): Output from calculate_feature_importance()
        top_n (int): Number of top features to display
    
    Returns:
        matplotlib.figure.Figure or None: Feature importance plot
    """
    if not feature_importance_dict['available']:
        print("Feature importance not available for this model type.")
        print("Note: k-NN does not provide feature importance.")
        print("Feature importance will be available in Iteration 3+ (Random Forest).")
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
    ax.invert_yaxis()  # Highest importance at top
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def print_advanced_metrics_summary(y_true, y_pred, y_pred_proba, model, 
                                   feature_names=None, stage_names=None):
    """
    Print a comprehensive summary of advanced metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_proba (np.ndarray): Predicted probabilities (if available)
        model: Trained model
        feature_names (list): Feature names (optional)
        stage_names (list): Sleep stage names (optional)
    """
    if stage_names is None:
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    
    print("\n" + "="*70)
    print("ADVANCED PERFORMANCE METRICS")
    print("="*70)
    
    # 1. Cohen's Kappa
    kappa = calculate_cohens_kappa(y_true, y_pred)
    kappa_interpretation = interpret_kappa(kappa)
    
    print("\n1. Cohen's Kappa (Inter-Rater Agreement)")
    print("-" * 70)
    print(f"   Kappa Score: {kappa:.4f}")
    print(f"   Interpretation: {kappa_interpretation}")
    print(f"   Note: Kappa accounts for chance agreement, making it more")
    print(f"         reliable than accuracy for imbalanced datasets.")
    
    # 2. ROC-AUC per class (only if probabilities are available)
    if y_pred_proba is not None and hasattr(model, 'predict_proba'):
        print("\n2. ROC-AUC Scores (One-vs-Rest)")
        print("-" * 70)
        
        roc_results = calculate_multiclass_roc_auc(y_true, y_pred_proba)
        
        print(f"   Per-Class AUC Scores:")
        for i, (stage_name, auc_score) in enumerate(zip(stage_names, roc_results['per_class'])):
            if not np.isnan(auc_score):
                print(f"      {stage_name:<8}: {auc_score:.4f}")
            else:
                print(f"      {stage_name:<8}: N/A (not in test set)")
        
        print(f"\n   Macro-Averaged AUC:    {roc_results['macro_avg']:.4f}")
        print(f"   Weighted-Averaged AUC: {roc_results['weighted_avg']:.4f}")
        print(f"   Note: AUC > 0.5 indicates better than random classification.")
        print(f"         AUC = 1.0 indicates perfect classification.")
    else:
        print("\n2. ROC-AUC Scores")
        print("-" * 70)
        print("   ⚠️  Not available: Model does not provide probability estimates.")
        print("   Note: k-NN can provide probabilities with predict_proba().")
        print("         Ensure you're using the probability output from the model.")
    
    # 3. Feature Importance
    print("\n3. Feature Importance Analysis")
    print("-" * 70)
    
    feature_importance = calculate_feature_importance(model, feature_names)
    
    if feature_importance['available']:
        importances = feature_importance['importances']
        sorted_indices = feature_importance['sorted_indices']
        
        print(f"   Top 10 Most Important Features:")
        for rank, idx in enumerate(sorted_indices[:10], 1):
            if feature_names is not None:
                feat_name = feature_names[idx]
            else:
                feat_name = f"Feature {idx}"
            print(f"      {rank:2d}. {feat_name:<30s}: {importances[idx]:.6f}")
        
        print(f"\n   Total features: {len(importances)}")
        print(f"   Features with >1% importance: {np.sum(importances > 0.01)}")
    else:
        print("   ⚠️  Not available for k-NN classifier (Iteration 1).")
        print("   Note: Feature importance will be available in Iteration 3+")
        print("         when using Random Forest classifier.")
        print("   Alternative: Consider using:")
        print("      - Permutation importance (sklearn.inspection.permutation_importance)")
        print("      - Univariate feature selection scores")
    
    print("\n" + "="*70)


# Helper function to generate feature names for Iteration 1
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