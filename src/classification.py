
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

# Import the new metrics module
try:
    from src.metrics import (
        calculate_cohens_kappa,
        interpret_kappa,
        calculate_multiclass_roc_auc,
        plot_roc_curves,
        calculate_feature_importance,
        plot_feature_importance,
        print_advanced_metrics_summary,
        generate_iteration1_feature_names
    )
    HAS_METRICS = True
except ImportError:
    print("⚠️  Warning: metrics.py not found. Advanced metrics will not be available.")
    HAS_METRICS = False


def train_classifier(features, labels, config):

    print(f"Training {config.CLASSIFIER_TYPE} classifier...")
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    # Basic validation
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("No features available for training!")

    # Helper function to create model based on iteration
    def create_model():
        """Create a fresh model instance based on current iteration."""
        if config.CURRENT_ITERATION == 1:
            # Iteration 1: Simple k-NN
            model = KNeighborsClassifier(n_neighbors=config.KNN_N_NEIGHBORS)
            return model, f"k-NN with k={config.KNN_N_NEIGHBORS}"

        elif config.CURRENT_ITERATION == 2:
            # Iteration 2: SVM
            model = SVC(
                C=getattr(config, 'SVM_C', 1.0),
                kernel=getattr(config, 'SVM_KERNEL', 'rbf'),
                probability=True,  # Enable probability estimates for ROC-AUC
                random_state=42
            )
            return model, f"SVM with C={model.C}, kernel={model.kernel}"

        elif config.CURRENT_ITERATION >= 3:
            # Iteration 3+: Random Forest
            model = RandomForestClassifier(
                n_estimators=getattr(config, 'RF_N_ESTIMATORS', 100),
                max_depth=getattr(config, 'RF_MAX_DEPTH', None),
                min_samples_split=getattr(config, 'RF_MIN_SAMPLES_SPLIT', 2),
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
            return model, f"Random Forest with {model.n_estimators} trees"

        else:
            raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    # Create initial model to display info
    _, model_description = create_model()
    print(f"Using {model_description}")

    # K-Fold Cross-Validation
    n_folds = config.CV_FOLDS
    print(f"\nPerforming {n_folds}-fold stratified cross-validation...")
    print("(Stratified to maintain class balance in each fold)")

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_f1_scores = []
    fold_kappa_scores = []  # NEW: Store kappa scores
    fold_predictions = []
    fold_true_labels = []
    fold_probabilities = []  # NEW: Store probabilities for ROC-AUC

    print("\nCross-Validation Results:")
    print("-" * 50)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(features, labels), 1):
        X_train_fold = features[train_idx]
        X_test_fold = features[test_idx]
        y_train_fold = labels[train_idx]
        y_test_fold = labels[test_idx]

        model_fold, _ = create_model()

        model_fold.fit(X_train_fold, y_train_fold)

        y_pred_fold = model_fold.predict(X_test_fold)
        fold_acc = accuracy_score(y_test_fold, y_pred_fold)
        fold_f1 = f1_score(y_test_fold, y_pred_fold, average='macro', zero_division=0)
        
        # NEW: Calculate Cohen's Kappa for this fold
        if HAS_METRICS:
            fold_kappa = calculate_cohens_kappa(y_test_fold, y_pred_fold)
            fold_kappa_scores.append(fold_kappa)
        else:
            fold_kappa = 0.0  # Placeholder if metrics not available

        fold_accuracies.append(fold_acc)
        fold_f1_scores.append(fold_f1)
        fold_predictions.extend(y_pred_fold)
        fold_true_labels.extend(y_test_fold)
        
        # NEW: Get probability predictions if available
        if hasattr(model_fold, 'predict_proba'):
            y_proba_fold = model_fold.predict_proba(X_test_fold)
            fold_probabilities.extend(y_proba_fold)

        print(f"Fold {fold_idx}/{n_folds}: Accuracy={fold_acc:.3f}, "
              f"Macro F1={fold_f1:.3f}, Kappa={fold_kappa:.3f}")

    print("-" * 50)

    # Display cross-validation summary statistics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    
    # NEW: Kappa summary
    if HAS_METRICS and len(fold_kappa_scores) > 0:
        mean_kappa = np.mean(fold_kappa_scores)
        std_kappa = np.std(fold_kappa_scores)
    else:
        mean_kappa = 0.0
        std_kappa = 0.0

    print(f"\nCross-Validation Summary:")
    print(f"Mean Accuracy: {mean_accuracy:.3f} (+/- {std_accuracy:.3f})")
    print(f"Mean Macro F1-Score: {mean_f1:.3f} (+/- {std_f1:.3f})")
    if HAS_METRICS:
        print(f"Mean Cohen's Kappa: {mean_kappa:.3f} (+/- {std_kappa:.3f})")

    print(f"Accuracy Range: [{min(fold_accuracies):.3f}, {max(fold_accuracies):.3f}]")

    # Display comprehensive performance metrics across all folds
    print("\nComprehensive Performance Metrics (Across All CV Folds):")
    print_performance_metrics(np.array(fold_true_labels), np.array(fold_predictions))

    # NEW: Display advanced metrics if available
    if HAS_METRICS:
        print("\n" + "="*70)
        print("ADVANCED METRICS ANALYSIS")
        print("="*70)
        
        # Generate feature names for Iteration 1
        n_channels = features.shape[1] // 16  # Assuming 16 features per channel
        feature_names = generate_iteration1_feature_names(n_channels=max(1, n_channels))
        
        # Ensure feature_names matches actual feature count
        if len(feature_names) != features.shape[1]:
            feature_names = [f'Feature_{i}' for i in range(features.shape[1])]
        
        # Convert probabilities to numpy array if available
        if len(fold_probabilities) > 0:
            fold_probabilities_array = np.array(fold_probabilities)
        else:
            fold_probabilities_array = None
        
        # Print advanced metrics summary
        print_advanced_metrics_summary(
            y_true=np.array(fold_true_labels),
            y_pred=np.array(fold_predictions),
            y_pred_proba=fold_probabilities_array,
            model=model_fold,  # Use last fold's model for feature importance check
            feature_names=feature_names,
            stage_names=['Wake', 'N1', 'N2', 'N3', 'REM']
        )
        
        # NEW: Plot ROC curves if probabilities available
        if fold_probabilities_array is not None:
            print("\nGenerating ROC curves...")
            roc_fig = plot_roc_curves(
                np.array(fold_true_labels),
                fold_probabilities_array,
                n_classes=5,
                stage_names=['Wake', 'N1', 'N2', 'N3', 'REM']
            )
            # Save figure
            roc_fig.savefig('roc_curves_cv.png', dpi=150, bbox_inches='tight')
            print(f"✓ ROC curves saved to: roc_curves_cv.png")
        
        # NEW: Plot feature importance if available
        print("\nGenerating feature importance plot...")
        feature_importance_dict = calculate_feature_importance(model_fold, feature_names)
        if feature_importance_dict['available']:
            fi_fig = plot_feature_importance(feature_importance_dict, top_n=20)
            if fi_fig is not None:
                fi_fig.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
                print(f"✓ Feature importance plot saved to: feature_importance.png")
        else:
            print("ℹ️  Feature importance not available for k-NN (Iteration 1)")
            print("   This will be available in Iteration 3+ with Random Forest")

    # Train final model on ALL data for deployment/prediction
    print("\n" + "="*70)
    print("Training final model on all available data...")
    print("="*70)
    final_model, _ = create_model()
    final_model.fit(features, labels)
    print("Final model training complete!")

    return final_model


def print_performance_metrics(y_true, y_pred):
    """
    Print comprehensive performance metrics for sleep stage classification.

    Includes accuracy, sensitivity (recall), specificity, and F1-score for each sleep stage.
    """

    # Sleep stage labels and names (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    stage_labels = list(range(5))

    print("\n" + "="*70)
    print("SLEEP STAGE CLASSIFICATION PERFORMANCE METRICS")
    print("="*70)

    # Overall metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print(f"Macro F1-Score: {macro_f1:.3f}")
    print(f"Weighted F1-Score: {weighted_f1:.3f}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=stage_labels)

    # Create a formatted confusion matrix
    cm_df = pd.DataFrame(cm, index=stage_names, columns=stage_names)
    print(cm_df.to_string())

    # Per-class metrics
    print("\nPer-Class Performance Metrics:")
    print("-" * 70)
    print(f"{'Stage':<8} {'Accuracy':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1-Score':<10}")
    print("-" * 70)

    # Calculate metrics for each sleep stage
    for i, stage_name in enumerate(stage_names):
        if i in y_true:  # Only calculate if stage is present in test set
            # Per-class accuracy (percentage of this class correctly classified)
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:
                class_accuracy = np.sum((y_pred == i) & (y_true == i)) / np.sum(class_mask)
            else:
                class_accuracy = 0.0

            # Sensitivity (Recall) - True Positive Rate
            sensitivity = recall_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]

            # Specificity - True Negative Rate
            tn = np.sum((y_true != i) & (y_pred != i))
            fp = np.sum((y_true != i) & (y_pred == i))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # F1-Score
            f1 = f1_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]

            print(f"{stage_name:<8} {class_accuracy:<10.3f} {sensitivity:<12.3f} {specificity:<12.3f} {f1:<10.3f}")
        else:
            print(f"{stage_name:<8} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10}")

    print("-" * 70)

    # Class distribution in test set
    print("\nClass Distribution in Test Set:")
    unique, counts = np.unique(y_true, return_counts=True)
    total_samples = len(y_true)

    for stage_idx, count in zip(unique, counts):
        stage_name = stage_names[stage_idx]
        percentage = count / total_samples * 100
        print(f"{stage_name}: {count} samples ({percentage:.1f}%)")

    # Sleep scoring specific notes
    print("\nNotes for Sleep Scoring:")
    print("- Sensitivity = Recall = True Positive Rate (correctly identified stages)")
    print("- Specificity = True Negative Rate (correctly rejected stages)")
    print("- Sleep stage imbalance is natural (more N2, less N1/REM)")
    print("- Consider Cohen's kappa for chance-corrected agreement")
    print("- Clinical focus: High sensitivity for REM and N3 stages")