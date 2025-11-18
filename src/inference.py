import numpy as np
import pandas as pd
import os

def make_inference(model, holdout_data, config):
    """
    Makes predictions on the hold-out data using the trained model.

    Args:
        model (object): The trained classification model.
        holdout_data (np.ndarray): The preprocessed and feature-extracted hold-out data.
        config (module): The configuration module.

    Returns:
        np.ndarray: Predicted labels for the hold-out data.
    """
    print("Making inference on hold-out data...")
    print(f"Holdout features shape: {holdout_data.shape}")
    print(f"Features contain NaN: {np.any(np.isnan(holdout_data))}")
    print(f"Features contain Inf: {np.any(np.isinf(holdout_data))}")

    print("\nHoldout feature statistics (BEFORE scaling):")
    print(f"  Global: min={np.min(holdout_data):.2e}, max={np.max(holdout_data):.2e}, mean={np.mean(holdout_data):.2e}")
    print(f"  Per-feature: mean range [{holdout_data.mean(axis=0).min():.2e}, {holdout_data.mean(axis=0).max():.2e}]")
    print(f"  Per-feature: std range [{holdout_data.std(axis=0).min():.2e}, {holdout_data.std(axis=0).max():.2e}]")

    # Find features with extreme values
    max_per_feature = np.max(np.abs(holdout_data), axis=0)
    extreme_features = np.where(max_per_feature > 1e10)[0]
    if len(extreme_features) > 0:
        print(f"\n⚠️  WARNING: {len(extreme_features)} features have extreme values (>1e10):")
        print(f"  Feature indices: {extreme_features[:10]}..." if len(extreme_features) > 10 else f"  Feature indices: {extreme_features}")
        print(f"  Max values: {max_per_feature[extreme_features[:5]]}")

    # Apply scaler if model has one (SVM with StandardScaler)
    if hasattr(model, 'scaler'):
        print("\nApplying StandardScaler to holdout data...")
        print(f"  Scaler was trained with mean range: [{model.scaler.mean_.min():.2e}, {model.scaler.mean_.max():.2e}]")
        print(f"  Scaler was trained with scale range: [{model.scaler.scale_.min():.2e}, {model.scaler.scale_.max():.2e}]")

        holdout_data = model.scaler.transform(holdout_data)

        print("\nHoldout feature statistics (AFTER scaling):")
        print(f"  Global: min={np.min(holdout_data):.2e}, max={np.max(holdout_data):.2e}, mean={np.mean(holdout_data):.2e}")

        # Check for features that are still very large after scaling
        max_scaled = np.max(np.abs(holdout_data), axis=0)
        outlier_features = np.where(max_scaled > 10)[0]
        if len(outlier_features) > 0:
            print(f"\n⚠️  WARNING: {len(outlier_features)} features have large scaled values (>10):")
            print(f"  This suggests distribution mismatch between training and holdout data!")
            print(f"  Feature indices: {outlier_features[:10]}..." if len(outlier_features) > 10 else f"  Feature indices: {outlier_features}")

    predictions = model.predict(holdout_data)

    # Print prediction distribution
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPrediction distribution:")
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    for stage, count in zip(unique, counts):
        print(f"  {stage_names[stage]}: {count} ({count/len(predictions)*100:.1f}%)")

    return predictions

def generate_submission_file(predictions, record_numbers, epoch_numbers, config):
    """
    Generates a submission CSV file.

    Args:
        predictions (np.ndarray): The predicted sleep stage labels.
        record_numbers (list): List of record numbers corresponding to each epoch.
        epoch_numbers (list): List of epoch numbers corresponding to each epoch.
        config (module): The configuration module.
    """
    print(f"Generating submission file: {config.SUBMISSION_FILE}...")
    submission_df = pd.DataFrame({
        'record_number': record_numbers,
        'epoch_number': epoch_numbers,
        'label': predictions
    })
    submission_df.to_csv(os.path.join(config.DATA_DIR, config.SUBMISSION_FILE), index=False)
    print("Submission file generated successfully.")
