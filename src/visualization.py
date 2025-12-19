"""
Visualization Module for Sleep Stage Classification

Provides plotting functions for model evaluation:
- Confusion matrices
- ROC curves
- Feature importance plots
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots and saves a confusion matrix.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        class_names (list): The names of the classes.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


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


def visualize_results(metrics_dict, config):
    """
    Visualizes the basic classification results using test set predictions.

    Args:
        metrics_dict (dict): Dictionary containing all metrics from training,
                             including y_true, y_pred from the test set.
        config (module): The configuration module.
    """
    print("Visualizing results...")
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

    # Use test set predictions from metrics_dict (not full dataset predictions)
    y_true = metrics_dict['y_true']
    y_pred = metrics_dict['y_pred']

    plot_confusion_matrix(y_true, y_pred, class_names)


def visualize_advanced_metrics(metrics_dict, config, selected_feature_names=None):
    """
    Generate all advanced visualizations from metrics dictionary.

    Creates ROC curves and feature importance plots based on available metrics.

    Args:
        metrics_dict (dict): Dictionary containing all metrics from training
        config: Configuration module
        selected_feature_names (list): List of selected feature names (optional)
    """
    # Import metrics calculation functions
    try:
        from src.metrics import calculate_feature_importance
        HAS_METRICS = True
    except ImportError:
        HAS_METRICS = False

    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

    # 1. Plot ROC curves if probabilities available
    if metrics_dict.get('y_pred_proba') is not None:
        roc_fig = plot_roc_curves(
            metrics_dict['y_true'],
            metrics_dict['y_pred_proba'],
            n_classes=5,
            stage_names=stage_names
        )
        if roc_fig is not None:
            roc_fig.savefig('roc_curves_cv.png', dpi=150, bbox_inches='tight')
            plt.close(roc_fig)

    # 2. Plot feature importance if available (Random Forest only)
    if HAS_METRICS and config.CURRENT_ITERATION >= 3:
        feature_importance_dict = calculate_feature_importance(
            metrics_dict.get('model'),
            feature_names=selected_feature_names  # Use actual feature names
        )

        if feature_importance_dict['available']:
            fi_fig = plot_feature_importance(feature_importance_dict, top_n=20)
            if fi_fig is not None:
                fi_fig.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
                plt.close(fi_fig)