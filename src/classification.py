import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

def train_classifier(features, labels, config):
    """
    STUDENT IMPLEMENTATION AREA: Train classifier based on iteration.

    This function provides a basic framework but students should enhance it:

    1. DONE: Implement proper cross-validation (not just train/test split)
    2. DONE: (use StratifiedKFold) Address class imbalance in sleep stage data
    3. Tune hyperparameters for each classifier
    4. Add more sophisticated evaluation metrics
    5. Consider ensemble methods in later iterations

    Args:
        features (np.ndarray): The input features.
        labels (np.ndarray): The corresponding labels.
        config (module): The configuration module.

    Returns:
        object: The trained classifier.
    """
    print(f"Training {config.CLASSIFIER_TYPE} classifier...")
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    # Basic validation
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("No features available for training!")

    # NOTE: Used k fold stratified to address class imbalance in sleep data:

    # Helper function to create model based on iteration
    def create_model():
        """Create a fresh model instance based on current iteration."""
        if config.CURRENT_ITERATION == 1:
            # Iteration 1: Simple k-NN
            model = KNeighborsClassifier(n_neighbors=config.KNN_N_NEIGHBORS)
            return model, f"k-NN with k={config.KNN_N_NEIGHBORS}"

        elif config.CURRENT_ITERATION == 2:
            # Iteration 2: SVM
            # TODO: Students should tune hyperparameters (C, kernel, gamma)
            model = SVC(
                C=getattr(config, 'SVM_C', 1.0),
                kernel=getattr(config, 'SVM_KERNEL', 'rbf'),
                random_state=42
            )
            return model, f"SVM with C={model.C}, kernel={model.kernel}"

        elif config.CURRENT_ITERATION >= 3:
            # Iteration 3+: Random Forest
            # TODO: Students should tune hyperparameters (n_estimators, max_depth, etc.)
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
    fold_predictions = []
    fold_true_labels = []

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

        fold_accuracies.append(fold_acc)
        fold_f1_scores.append(fold_f1)
        fold_predictions.extend(y_pred_fold)
        fold_true_labels.extend(y_test_fold)

        print(f"Fold {fold_idx}/{n_folds}: Accuracy={fold_acc:.3f}, Macro F1={fold_f1:.3f}")

    print("-" * 50)

    # Display cross-validation summary statistics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)

    print(f"\nCross-Validation Summary:")
    print(f"Mean Accuracy: {mean_accuracy:.3f} (+/- {std_accuracy:.3f})")
    print(f"Mean Macro F1-Score: {mean_f1:.3f} (+/- {std_f1:.3f})")
    print(f"Accuracy Range: [{min(fold_accuracies):.3f}, {max(fold_accuracies):.3f}]")

    # Display comprehensive performance metrics across all folds
    print("\nComprehensive Performance Metrics (Across All CV Folds):")
    print_performance_metrics(np.array(fold_true_labels), np.array(fold_predictions))

    # Train final model on ALL data for deployment/prediction
    print("\n" + "="*70)
    print("Training final model on all available data...")
    print("="*70)
    final_model, _ = create_model()
    final_model.fit(features, labels)
    print("Final model training complete!")

    # TODO: Students should add more advanced metrics:
    # - Cohen's kappa (important for sleep scoring)
    # - ROC-AUC for each class
    # - Feature importance analysis
    print("\nTODO: Students should add Cohen's kappa and ROC-AUC metrics")

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
