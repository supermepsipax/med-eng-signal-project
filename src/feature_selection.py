import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif

def select_features(features, labels, config, selector_info=None):
    """
    2-stage feature selection for Iteration 3+.

    Strategy (based on PROJECT_GUIDE.md lines 1768-1803):
    1. Correlation Analysis: Remove highly correlated features (r > 0.95)
    2. Statistical Testing: Select top-k using mutual information

    Note: Variance thresholding removed due to scale mismatch issues between
    normalized frequency features (0-1 range) and raw time features.

    Target: Reduce from ~64 features to 40 features

    Args:
        features (np.ndarray): The input features (n_samples, n_features).
        labels (np.ndarray): The corresponding labels (None during inference).
        config (module): The configuration module.
        selector_info (dict): Pre-fitted selectors for inference (None during training).

    Returns:
        tuple: (selected_features, selector_info) where:
            - selected_features: Transformed features
            - selector_info: Dict with selectors (None for iterations 1-2, dict for 3+)
    """
    print(f"Selecting features for iteration {config.CURRENT_ITERATION}...")
    print(f"Input features shape: {features.shape}")

    if features.shape[1] == 0:
        print("⚠️  WARNING: No features to select from!")
        return features, None

    if config.CURRENT_ITERATION <= 2:
        # Early iterations: Use all available features
        print("Early iteration - using all available features")
        selected_features = features
        return selected_features, None

    elif config.CURRENT_ITERATION >= 3:
        # Check if we're in inference mode (selector_info provided)
        if selector_info is not None:
            print("\n" + "="*70)
            print("APPLYING PRE-FITTED FEATURE SELECTION (INFERENCE MODE)")
            print("="*70)
            return apply_feature_selection(features, selector_info)

        # Training mode: Fit new selectors
        # Iteration 3+: 2-stage feature selection (correlation + mutual information)
        print("\n" + "="*70)
        print("FEATURE SELECTION - 2-STAGE APPROACH")
        print("="*70)
        print(f"Input: {features.shape[1]} features")

        # Stage 1: Correlation Analysis (remove redundant features)
        print("\nStage 1: Correlation Analysis")
        print(f"  Input: {features.shape[1]} features")

        if features.shape[1] > 1:
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(features, rowvar=False)

            # Find highly correlated feature pairs (r > 0.95)
            high_corr_pairs = np.where(np.abs(corr_matrix) > 0.95)
            high_corr_pairs = [(i, j) for i, j in zip(*high_corr_pairs) if i < j]

            # Remove one feature from each correlated pair
            features_to_remove = set()
            for i, j in high_corr_pairs:
                if i not in features_to_remove:
                    features_to_remove.add(j)

            features_to_keep = [i for i in range(features.shape[1])
                               if i not in features_to_remove]
            features_stage1 = features[:, features_to_keep]

            print(f"  Removed {len(features_to_remove)} highly correlated features (r > 0.95)")
        else:
            features_stage1 = features
            print("  Skipping correlation analysis (too few features)")

        print(f"  Output: {features_stage1.shape[1]} features")

        # Stage 2: Mutual Information Selection
        print("\nStage 2: Mutual Information Selection")
        print(f"  Input: {features_stage1.shape[1]} features")

        # Select top-k features using mutual information
        target_k = getattr(config, 'FEATURE_SELECTION_K', 40)
        k = min(target_k, features_stage1.shape[1])

        if k < features_stage1.shape[1]:
            mi_selector = SelectKBest(mutual_info_classif, k=k)
            selected_features = mi_selector.fit_transform(features_stage1, labels)
            print(f"  Selected top {k} features based on mutual information")
        else:
            selected_features = features_stage1
            mi_selector = None
            print(f"  Keeping all {features_stage1.shape[1]} features (less than target k={target_k})")

        print(f"  Output: {selected_features.shape[1]} features")

        print("\n" + "="*70)
        print(f"FINAL: Reduced from {features.shape[1]} to {selected_features.shape[1]} features")
        print("="*70)

        # Package selectors for inference (no variance selector)
        selector_info = {
            'variance_selector': None,  # Removed variance thresholding
            'features_to_keep_after_corr': features_to_keep if features.shape[1] > 1 else None,
            'mi_selector': mi_selector,
            'n_input_features': features.shape[1],
            'n_output_features': selected_features.shape[1]
        }

        print(f"\n✓ Selector info saved for inference")
        print(f"  - Variance selector: not used (removed)")
        print(f"  - Correlation filter: {'fitted' if selector_info['features_to_keep_after_corr'] else 'not used'}")
        print(f"  - MI selector: {'fitted' if selector_info['mi_selector'] else 'not used'}")

        return selected_features, selector_info

    else:
        # Should not reach here, but handle gracefully
        selected_features = features
        return selected_features, None


def apply_feature_selection(features, selector_info):
    """
    Apply pre-fitted feature selectors to new data (inference mode).

    This ensures the exact same features selected during training are used
    during inference on holdout data.

    Args:
        features (np.ndarray): Input features from holdout data
        selector_info (dict): Dictionary containing fitted selectors from training

    Returns:
        tuple: (selected_features, selector_info)
    """
    print(f"  Input features: {features.shape[1]}")
    print(f"  Expected output: {selector_info['n_output_features']} features")

    # Validate input feature count
    if features.shape[1] != selector_info['n_input_features']:
        raise ValueError(
            f"Feature count mismatch! Training used {selector_info['n_input_features']} features, "
            f"but inference has {features.shape[1]} features. "
            f"Ensure feature extraction is identical between training and inference."
        )

    current_features = features

    # Stage 1: Apply variance selector
    if selector_info['variance_selector'] is not None:
        print("\n  Stage 1: Applying variance threshold...")
        current_features = selector_info['variance_selector'].transform(current_features)
        print(f"    → {current_features.shape[1]} features")
    else:
        print("\n  Stage 1: Variance threshold not used during training (skipped)")

    # Stage 2: Apply correlation filter
    if selector_info['features_to_keep_after_corr'] is not None:
        print("  Stage 2: Applying correlation filter...")
        current_features = current_features[:, selector_info['features_to_keep_after_corr']]
        print(f"    → {current_features.shape[1]} features")
    else:
        print("  Stage 2: Correlation filter not used during training (skipped)")

    # Stage 3: Apply mutual information selector
    if selector_info['mi_selector'] is not None:
        print("  Stage 3: Applying mutual information selector...")
        current_features = selector_info['mi_selector'].transform(current_features)
        print(f"    → {current_features.shape[1]} features")
    else:
        print("  Stage 3: MI selector not used during training (skipped)")

    # Validate output feature count
    if current_features.shape[1] != selector_info['n_output_features']:
        raise ValueError(
            f"Output feature count mismatch! Expected {selector_info['n_output_features']} "
            f"but got {current_features.shape[1]}. Feature selection pipeline error."
        )

    print(f"\n  ✓ Feature selection applied: {features.shape[1]} → {current_features.shape[1]} features")
    print("="*70)

    return current_features, selector_info
