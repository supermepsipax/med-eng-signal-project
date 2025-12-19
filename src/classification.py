import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import RobustScaler
import pandas as pd

# Import the new metrics module
try:
    from src.metrics import calculate_cohens_kappa

    HAS_METRICS = True
except ImportError:
    print("⚠️  Warning: metrics.py not found. Advanced metrics will not be available.")
    HAS_METRICS = False


def train_classifier(features, labels, config, record_ids=None):
    """
    Train a classifier with iteration-specific strategies.

    Iteration 1 (k-NN): Simple CV evaluation, optional k tuning
    Iteration 2 (SVM): GridSearchCV for hyperparameter tuning (C, gamma, kernel)
    Iteration 3+ (RF): GridSearchCV for hyperparameter tuning (n_estimators, max_depth, etc.)
                       with subject-wise cross-validation (GroupKFold)

    Uses proper train/test split to avoid data leakage and make CV useful for model improvement.

    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Labels array (n_samples,)
        config: Configuration module
        record_ids: Array of record IDs for subject-wise splitting (optional but recommended for Iteration 3+)
    """
    print(f"Training classifier for Iteration {config.CURRENT_ITERATION}...")
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("No features available for training!")

    print("\n" + "=" * 70)
    print("DATA SPLIT")
    print("=" * 70)

    # For Iteration 3+, warn if record_ids not provided
    if config.CURRENT_ITERATION >= 3 and record_ids is None:
        print("⚠️  WARNING: record_ids not provided for Iteration 3+")
        print("   GridSearchCV will use standard StratifiedKFold (may overfit to subjects)")
        print("   For proper subject-independent evaluation, pass record_ids to train_classifier()")

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Also split record_ids if provided
    if record_ids is not None:
        _, _, record_ids_train, _ = train_test_split(
            features, record_ids, test_size=0.2, random_state=42, stratify=labels
        )
    print(
        f"Training set: {X_train.shape[0]} samples ({X_train.shape[0] / features.shape[0] * 100:.1f}%)"
    )
    print(
        f"Test set: {X_test.shape[0]} samples ({X_test.shape[0] / features.shape[0] * 100:.1f}%)"
    )

    train_classes, train_counts = np.unique(y_train, return_counts=True)
    test_classes, test_counts = np.unique(y_test, return_counts=True)
    stage_names = ["Wake", "N1", "N2", "N3", "REM"]
    print("\nClass distribution (train/test):")
    for cls in range(5):
        if cls in train_classes and cls in test_classes:
            train_pct = train_counts[train_classes == cls][0] / len(y_train) * 100
            test_pct = test_counts[test_classes == cls][0] / len(y_test) * 100
            print(f"  {stage_names[cls]}: {train_pct:.1f}% / {test_pct:.1f}%")

    if config.CURRENT_ITERATION == 1:
        best_model = train_knn(X_train, y_train, config)
    elif config.CURRENT_ITERATION == 2:
        best_model = train_svm(X_train, y_train, config)
    elif config.CURRENT_ITERATION >= 3:
        # Pass record_ids to enable subject-wise cross-validation
        groups_train = record_ids_train if record_ids is not None else None
        best_model = train_random_forest(X_train, y_train, config, groups=groups_train)
    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    # Evaluate on held-out test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 70)

    # Apply scaler if model has one (SVM with RobustScaler)
    if hasattr(best_model, "scaler"):
        print("Applying RobustScaler to test set...")
        X_test_scaled = best_model.scaler.transform(X_test)
        y_pred = best_model.predict(X_test_scaled)
        if hasattr(best_model, "predict_proba"):
            y_pred_proba = best_model.predict_proba(X_test_scaled)
    else:
        y_pred = best_model.predict(X_test)
        y_pred_proba = None
        if hasattr(best_model, "predict_proba"):
            y_pred_proba = best_model.predict_proba(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Calculate Cohen's Kappa
    if HAS_METRICS:
        test_kappa = calculate_cohens_kappa(y_test, y_pred)
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"Test Macro F1: {test_f1:.3f}")
        print(f"Test Cohen's Kappa: {test_kappa:.3f}")
    else:
        test_kappa = None
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"Test Macro F1: {test_f1:.3f}")

    # Calculate per-class metrics
    per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Build comprehensive metrics dictionary
    metrics_dict = {
        # Test set metrics
        "test_accuracy": test_accuracy,
        "test_f1_macro": test_f1,
        "test_f1_weighted": test_weighted_f1,
        "test_kappa": test_kappa,
        # Predictions and ground truth
        "y_true": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        # Detailed metrics
        "confusion_matrix": conf_matrix,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        # Metadata
        "stage_names": stage_names,
        "n_train_samples": len(y_train),
        "n_test_samples": len(y_test),
        "n_features": features.shape[1],
        "n_samples": features.shape[0],  # Total samples (train + test)
        "classifier_type": config.CLASSIFIER_TYPE,
        "iteration": config.CURRENT_ITERATION,
        "model": best_model,  # Include model for feature importance in visualization
        # CV results format for report.py compatibility (maps test metrics to CV format)
        "cv_results": {
            "mean_accuracy": test_accuracy,
            "std_accuracy": 0.0,  # N/A for train/test split
            "mean_f1": test_f1,
            "std_f1": 0.0,  # N/A for train/test split
            "mean_kappa": test_kappa if test_kappa is not None else 0.0,
            "std_kappa": 0.0,  # N/A for train/test split
        },
    }

    print_performance_metrics(y_test, y_pred)

    return best_model, metrics_dict


def train_knn(X_train, y_train, config):
    """
    Train k-NN classifier with optional k tuning via cross-validation.

    Strategy: k-NN is simple and has minimal hyperparameters. We can optionally
    tune k using CV, but the default k=5 often works well.
    """
    print("\n" + "=" * 70)
    print("ITERATION 1: k-NN CLASSIFIER")
    print("=" * 70)

    # Option 1: Use fixed k (faster, simpler)
    use_tuning = getattr(config, "KNN_TUNE_K", False)

    if not use_tuning:
        k = getattr(config, "KNN_N_NEIGHBORS", 5)
        print(f"Using fixed k={k}")
        print(f"  (Set KNN_TUNE_K=True in config to enable hyperparameter tuning)")

        model = KNeighborsClassifier(n_neighbors=k)

        # Train on full training set
        print(f"\nTraining model on full training set...")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  k (neighbors): {k}")
        model.fit(X_train, y_train)
        print("✓ Training complete")

        return model

    else:
        # Option 2: Tune k using GridSearchCV (slower but finds optimal k)
        print("Tuning k parameter using GridSearchCV...")
        k_range = range(1, min(31, len(X_train) // 10))  # Test k from 1 to 30
        param_grid = {"n_neighbors": k_range}

        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=config.CV_FOLDS,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        print(f"\nBest k: {grid_search.best_params_['n_neighbors']}")
        print(f"Best CV Accuracy: {grid_search.best_score_:.3f}")

        return grid_search.best_estimator_


def train_svm(X_train, y_train, config):
    """
    Train SVM classifier with optional hyperparameter tuning.

    If SVM_TUNE_HYPERPARAMS=False: Train single model with fixed parameters (fast)
    If SVM_TUNE_HYPERPARAMS=True: Use GridSearchCV for hyperparameter tuning (slow but optimal)
    """
    print("\n" + "=" * 70)
    print("ITERATION 2: SVM CLASSIFIER")
    print("=" * 70)

    # Check if hyperparameter tuning is enabled
    use_tuning = getattr(config, "SVM_TUNE_HYPERPARAMS", False)

    if not use_tuning:
        # Option 1: Train single model with fixed parameters (FAST)
        C = getattr(config, "SVM_C", 1.0)
        kernel = getattr(config, "SVM_KERNEL", "rbf")
        gamma = getattr(config, "SVM_GAMMA", "scale")

        print(f"Training single SVM model with fixed parameters:")
        print(f"  C={C}, kernel='{kernel}', gamma='{gamma}', class_weight='balanced'")
        print(
            f"  (Set SVM_TUNE_HYPERPARAMS=True in config to enable hyperparameter tuning)"
        )

        # CRITICAL: Scale features for SVM (using RobustScaler for domain robustness)
        print("\n⚙️  Applying RobustScaler (more robust to outliers and domain shift)...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        print("   Training feature statistics (BEFORE scaling):")
        print(
            f"     Global: min={X_train.min():.2e}, max={X_train.max():.2e}, mean={X_train.mean():.2e}"
        )
        print(
            f"     Per-feature: mean range [{X_train.mean(axis=0).min():.2e}, {X_train.mean(axis=0).max():.2e}]"
        )
        print(
            f"     Per-feature: std range  [{X_train.std(axis=0).min():.2e}, {X_train.std(axis=0).max():.2e}]"
        )
        print("   Training feature statistics (AFTER scaling):")
        print(
            f"     Global: min={X_train_scaled.min():.2e}, max={X_train_scaled.max():.2e}, mean={X_train_scaled.mean():.2e}"
        )
        print(f"   Scaler learned from {X_train.shape[0]} training samples:")
        print(f"     Scaler mean range: [{scaler.mean_.min():.2e}, {scaler.mean_.max():.2e}]")
        print(f"     Scaler scale range: [{scaler.scale_.min():.2e}, {scaler.scale_.max():.2e}]")

        # Use class_weight='balanced' to handle class imbalance
        model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight="balanced",  # CRITICAL for imbalanced sleep data
            probability=True,
            random_state=42,
        )

        # Train on full training set with scaled features
        print("\nTraining model on full training set...")
        print(f"  Training samples: {X_train_scaled.shape[0]}")
        print(f"  Features: {X_train_scaled.shape[1]}")
        model.fit(X_train_scaled, y_train)
        print("✓ Training complete")

        # Store scaler with model for later use
        model.scaler = scaler

        return model

    else:
        # Option 2: Hyperparameter tuning with GridSearchCV (SLOW but optimal)
        print("Hyperparameter tuning enabled (this may take a while...)")

        # CRITICAL: Scale features for SVM (using RobustScaler for domain robustness)
        print("\n⚙️  Applying RobustScaler (more robust to outliers and domain shift)...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        print("   Training feature statistics (BEFORE scaling):")
        print(
            f"     Global: min={X_train.min():.2e}, max={X_train.max():.2e}, mean={X_train.mean():.2e}"
        )
        print("   Training feature statistics (AFTER scaling):")
        print(
            f"     Global: min={X_train_scaled.min():.2e}, max={X_train_scaled.max():.2e}, mean={X_train_scaled.mean():.2e}"
        )
        print(f"   Scaler learned from {X_train.shape[0]} training samples:")
        print(f"     Scaler mean range: [{scaler.mean_.min():.2e}, {scaler.mean_.max():.2e}]")
        print(f"     Scaler scale range: [{scaler.scale_.min():.2e}, {scaler.scale_.max():.2e}]")

        # Define hyperparameter search space
        param_grid = {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "kernel": ["rbf", "linear"],
        }

        # Allow config to override search space
        if hasattr(config, "SVM_PARAM_GRID"):
            param_grid = config.SVM_PARAM_GRID

        print("\nHyperparameter search space:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")

        print(f"\nPerforming GridSearchCV with {config.CV_FOLDS}-fold CV...")
        print(
            f"Total combinations to test: {np.prod([len(v) for v in param_grid.values()])}"
        )
        print("(This may take several minutes...)")

        grid_search = GridSearchCV(
            SVC(probability=True, class_weight="balanced", random_state=42),
            param_grid,
            cv=config.CV_FOLDS,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train_scaled, y_train)

        print("\n" + "-" * 70)
        print("GridSearchCV Results:")
        print("-" * 70)
        print(f"Best hyperparameters: {grid_search.best_params_}")
        print(f"Best CV Accuracy: {grid_search.best_score_:.3f}")

        # Show top 5 configurations
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df = results_df.sort_values("rank_test_score")
        print("\nTop 5 configurations:")
        for i, row in results_df.head(5).iterrows():
            print(
                f"  Rank {int(row['rank_test_score'])}: "
                f"C={row['param_C']}, gamma={row['param_gamma']}, kernel={row['param_kernel']}, "
                f"Score={row['mean_test_score']:.3f} (+/- {row['std_test_score']:.3f})"
            )

        # Store scaler with model for later use
        best_model = grid_search.best_estimator_
        best_model.scaler = scaler

        return best_model


def train_random_forest(X_train, y_train, config, groups=None):
    """
    Train Random Forest classifier with GridSearchCV for hyperparameter tuning.

    Strategy: RF has many hyperparameters. Use GridSearchCV to tune the most
    important ones: n_estimators, max_depth, min_samples_split.

    For proper subject-independent evaluation, uses GroupKFold when groups are provided.

    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration module
        groups: Array of group labels (record IDs) for subject-wise CV (optional)
    """
    print("\n" + "=" * 70)
    print("ITERATION 3+: RANDOM FOREST CLASSIFIER WITH HYPERPARAMETER TUNING")
    print("=" * 70)

    # Define hyperparameter search space
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Allow config to override search space
    if hasattr(config, "RF_PARAM_GRID"):
        param_grid = config.RF_PARAM_GRID

    print("Hyperparameter search space:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    # Configure cross-validation strategy based on whether groups are provided
    if groups is not None:
        # Subject-wise cross-validation (recommended for sleep staging)
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        n_folds = min(config.CV_FOLDS, n_groups)  # Can't have more folds than groups

        cv_strategy = GroupKFold(n_splits=n_folds)
        cv_description = f"GroupKFold (subject-wise, {n_folds} folds)"

        print(f"\n🔬 Cross-Validation Strategy: {cv_description}")
        print(f"   Number of unique subjects: {n_groups}")
        print(f"   Number of folds: {n_folds}")
        print(f"   ✓ Subject-independent evaluation (no data leakage)")
        print(f"   Each fold trains on {n_folds-1} subjects, validates on 1 subject")
    else:
        # Standard stratified cross-validation (fallback)
        cv_strategy = config.CV_FOLDS
        cv_description = f"StratifiedKFold ({config.CV_FOLDS} folds)"

        print(f"\n⚠️  Cross-Validation Strategy: {cv_description}")
        print(f"   WARNING: Standard CV may have data leakage across subjects")
        print(f"   Recommendation: Pass record_ids to enable subject-wise CV")

    print(f"\nPerforming GridSearchCV with {cv_description}...")
    print(
        f"Total combinations to test: {np.prod([len(v) for v in param_grid.values()])}"
    )
    print("(This may take several minutes...)")

    grid_search = GridSearchCV(
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # CRITICAL: Handle class imbalance (N1 is only 2.7%!)
        ),
        param_grid,
        cv=cv_strategy,
        scoring="f1_macro",  # Optimize for F1 Macro (leaderboard metric!)
        n_jobs=-1,
        verbose=1,
    )

    # Fit with or without groups parameter
    if groups is not None:
        grid_search.fit(X_train, y_train, groups=groups)
    else:
        grid_search.fit(X_train, y_train)

    print("\n" + "-" * 70)
    print("GridSearchCV Results:")
    print("-" * 70)
    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Best CV F1 Macro: {grid_search.best_score_:.3f}")

    # Show top 5 configurations
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values("rank_test_score")
    print("\nTop 5 configurations:")
    for i, row in results_df.head(5).iterrows():
        print(
            f"  Rank {int(row['rank_test_score'])}: "
            f"n_est={row['param_n_estimators']}, max_depth={row['param_max_depth']}, "
            f"min_split={row['param_min_samples_split']}, min_leaf={row['param_min_samples_leaf']}, "
            f"Score={row['mean_test_score']:.3f} (+/- {row['std_test_score']:.3f})"
        )

    # Optional: Show feature importances from best model
    best_model = grid_search.best_estimator_
    print(
        f"\nBest model has {best_model.n_estimators} trees with "
        f"max_depth={best_model.max_depth}"
    )

    return best_model


def print_performance_metrics(y_true, y_pred):
    """
    Print comprehensive performance metrics for sleep stage classification.

    Includes accuracy, sensitivity (recall), specificity, and F1-score for each sleep stage.
    """
    # Sleep stage labels and names (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
    stage_names = ["Wake", "N1", "N2", "N3", "REM"]
    stage_labels = list(range(5))

    print("\n" + "=" * 70)
    print("SLEEP STAGE CLASSIFICATION PERFORMANCE METRICS")
    print("=" * 70)

    # Overall metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

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
    print(
        f"{'Stage':<8} {'Accuracy':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1-Score':<10}"
    )
    print("-" * 70)

    # Calculate metrics for each sleep stage
    for i, stage_name in enumerate(stage_names):
        if i in y_true:  # Only calculate if stage is present in test set
            # Per-class accuracy (percentage of this class correctly classified)
            class_mask = y_true == i
            if np.sum(class_mask) > 0:
                class_accuracy = np.sum((y_pred == i) & (y_true == i)) / np.sum(
                    class_mask
                )
            else:
                class_accuracy = 0.0

            # Sensitivity (Recall) - True Positive Rate
            sensitivity = recall_score(
                y_true, y_pred, labels=[i], average=None, zero_division=0
            )[0]

            # Specificity - True Negative Rate
            tn = np.sum((y_true != i) & (y_pred != i))
            fp = np.sum((y_true != i) & (y_pred == i))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # F1-Score
            f1 = f1_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]

            print(
                f"{stage_name:<8} {class_accuracy:<10.3f} {sensitivity:<12.3f} {specificity:<12.3f} {f1:<10.3f}"
            )
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
