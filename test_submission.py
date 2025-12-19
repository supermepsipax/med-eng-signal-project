"""
Test Submission - Sanity Check Script

This script evaluates a trained model on a held-out patient that was completely
excluded from training. This provides an independent verification of model
performance to validate leaderboard results.

Usage:
    1. Set EXCLUDE_PATIENTS_FOR_TESTING = ['R10'] in config.py
    2. Run main.py to train model without R10
    3. Set TEST_PATIENT = 'R10' below
    4. Run: python test_submission.py

The script will:
    - Load the held-out patient data WITH labels (from training directory)
    - Load the saved model, scaler, and feature selector
    - Run the complete preprocessing and feature extraction pipeline
    - Make predictions and calculate all metrics
    - Display confusion matrix, F1 scores, and per-class performance
"""

import config
from src.data_loader import load_training_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.feature_selection import apply_feature_selection
from src.utils import load_cache
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, cohen_kappa_score,
                             classification_report)
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
# Which patient to test on (must match EXCLUDE_PATIENTS_FOR_TESTING in config.py)
TEST_PATIENT = 'R10'

# Ensure this matches the iteration you trained
ITERATION = config.CURRENT_ITERATION

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("="*70)
    print("INDEPENDENT HOLDOUT PATIENT TEST")
    print("="*70)
    print(f"Testing patient: {TEST_PATIENT}")
    print(f"Using model from Iteration {ITERATION}")
    print("="*70)

    # Verify the patient was excluded from training
    if config.EXCLUDE_PATIENTS_FOR_TESTING is None:
        print("\n⚠️  WARNING: EXCLUDE_PATIENTS_FOR_TESTING is None in config.py!")
        print(f"   You must set EXCLUDE_PATIENTS_FOR_TESTING = ['{TEST_PATIENT}'] and retrain!")
        response = input("\nDid you already train with this patient excluded? (yes/no): ")
        if response.lower() != 'yes':
            print("Please update config.py, retrain with main.py, then run this script again.")
            return
    elif TEST_PATIENT not in config.EXCLUDE_PATIENTS_FOR_TESTING:
        print(f"\n⚠️  WARNING: {TEST_PATIENT} is NOT in EXCLUDE_PATIENTS_FOR_TESTING!")
        print(f"   Current EXCLUDE_PATIENTS_FOR_TESTING = {config.EXCLUDE_PATIENTS_FOR_TESTING}")
        print(f"   Update config.py and retrain before running this test.")
        return

    # Step 1: Load the held-out patient data WITH labels
    print(f"\n{'='*70}")
    print("STEP 1: Loading Held-Out Patient Data")
    print(f"{'='*70}")

    edf_file = os.path.join(config.TRAINING_DIR, f"{TEST_PATIENT}.edf")
    xml_file = os.path.join(config.TRAINING_DIR, f"{TEST_PATIENT}.xml")

    if not os.path.exists(edf_file):
        print(f"ERROR: EDF file not found: {edf_file}")
        return
    if not os.path.exists(xml_file):
        print(f"ERROR: XML file not found: {xml_file}")
        return

    # Load patient data with labels
    multi_channel_data, true_labels, channel_info = load_training_data(
        edf_file, xml_file, epoch_length=30
    )

    print(f"\n✓ Loaded {TEST_PATIENT}:")
    print(f"  Total epochs: {len(true_labels)}")
    print(f"  Duration: {len(true_labels)*30/3600:.2f} hours")

    # Create record_ids array for this single patient
    record_ids = np.array([TEST_PATIENT] * len(true_labels))

    # Step 2: Load the trained model
    print(f"\n{'='*70}")
    print("STEP 2: Loading Trained Model and Components")
    print(f"{'='*70}")

    model_filename = f"model_iter{ITERATION}.joblib"
    selector_filename = f"selector_info_iter{ITERATION}.joblib"

    model = load_cache(model_filename, config.CACHE_DIR)
    selector_info = load_cache(selector_filename, config.CACHE_DIR)

    if model is None:
        print(f"ERROR: Model file not found: {os.path.join(config.CACHE_DIR, model_filename)}")
        print("Please train the model first by running main.py")
        return

    print(f"✓ Loaded model: {config.CLASSIFIER_TYPE}")

    if selector_info is not None:
        print(f"✓ Loaded feature selector")
        print(f"  Selected features: {len(selector_info.get('selected_feature_names', []))}")
    else:
        print("  No feature selector (using all features)")

    # Step 3: Preprocess the data
    print(f"\n{'='*70}")
    print("STEP 3: Preprocessing")
    print(f"{'='*70}")

    preprocessed_data = preprocess(multi_channel_data, channel_info, config, record_ids)

    if isinstance(preprocessed_data, dict):
        print("Preprocessed multi-channel data:")
        for signal_type in preprocessed_data.keys():
            print(f"  {signal_type.upper()}: {preprocessed_data[signal_type].shape}")
    else:
        print(f"Preprocessed data shape: {preprocessed_data.shape}")

    # Step 4: Extract features
    print(f"\n{'='*70}")
    print("STEP 4: Feature Extraction")
    print(f"{'='*70}")

    features, feature_names = extract_features(preprocessed_data, config, channel_info)
    print(f"Extracted features shape: {features.shape}")
    print(f"Feature names: {len(feature_names)} features")

    # Step 5: Apply feature selection (if used during training)
    print(f"\n{'='*70}")
    print("STEP 5: Feature Selection")
    print(f"{'='*70}")

    if selector_info is not None:
        selected_features, _, _ = apply_feature_selection(features, selector_info)
        print(f"Selected features shape: {selected_features.shape}")
    else:
        selected_features = features
        print("No feature selection applied (using all features)")

    # Step 6: Make predictions
    print(f"\n{'='*70}")
    print("STEP 6: Making Predictions")
    print(f"{'='*70}")

    # Apply scaler if the model has one (for SVM)
    if hasattr(model, 'scaler'):
        print("Applying StandardScaler...")
        selected_features = model.scaler.transform(selected_features)

    predictions = model.predict(selected_features)

    print(f"✓ Generated {len(predictions)} predictions")

    # Step 7: Calculate metrics
    print(f"\n{'='*70}")
    print("STEP 7: RESULTS - Independent Holdout Patient Evaluation")
    print(f"{'='*70}")

    # Overall metrics
    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    weighted_f1 = f1_score(true_labels, predictions, average='weighted')
    kappa = cohen_kappa_score(true_labels, predictions)

    print(f"\n📊 OVERALL METRICS (Patient {TEST_PATIENT}):")
    print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Macro F1:    {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"  Weighted F1: {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
    print(f"  Cohen's κ:   {kappa:.4f}")

    # Per-class metrics
    print(f"\n📋 PER-CLASS METRICS:")
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

    precision = precision_score(true_labels, predictions, average=None, zero_division=0)
    recall = recall_score(true_labels, predictions, average=None, zero_division=0)
    f1_per_class = f1_score(true_labels, predictions, average=None, zero_division=0)

    print(f"\n{'Stage':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 50)
    for i, stage_name in enumerate(stage_names):
        if i < len(precision):
            print(f"{stage_name:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1_per_class[i]:<12.4f}")

    # Confusion matrix
    print(f"\n📈 CONFUSION MATRIX:")
    cm = confusion_matrix(true_labels, predictions)

    # Print header
    print(f"\n{'Actual \\ Pred':<15}", end="")
    for stage_name in stage_names:
        print(f"{stage_name:<10}", end="")
    print("\n" + "-" * 70)

    # Print each row
    for i, stage_name in enumerate(stage_names):
        if i < len(cm):
            print(f"{stage_name:<15}", end="")
            for j in range(len(cm[i])):
                print(f"{cm[i][j]:<10}", end="")
            print()

    # Class distribution comparison
    print(f"\n📊 CLASS DISTRIBUTION:")
    print(f"\n{'Stage':<10} {'True Count':<12} {'Pred Count':<12} {'True %':<10} {'Pred %':<10}")
    print("-" * 60)

    true_counts = np.bincount(true_labels, minlength=5)
    pred_counts = np.bincount(predictions, minlength=5)

    for i, stage_name in enumerate(stage_names):
        true_pct = (true_counts[i] / len(true_labels)) * 100
        pred_pct = (pred_counts[i] / len(predictions)) * 100
        print(f"{stage_name:<10} {true_counts[i]:<12} {pred_counts[i]:<12} "
              f"{true_pct:<10.1f} {pred_pct:<10.1f}")

    # Detailed classification report
    print(f"\n📝 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(true_labels, predictions,
                                target_names=stage_names,
                                digits=4,
                                zero_division=0))

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nThis is the TRUE performance of your model on completely unseen data.")
    print(f"\nKey metric for leaderboard comparison:")
    print(f"  ► MACRO F1 SCORE: {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"\nIf this is significantly different from your leaderboard score (~{25:.1f}%),")
    print(f"it suggests there may be an issue with the leaderboard evaluation method.")
    print(f"\nNext steps:")
    print(f"  1. Compare this {macro_f1*100:.2f}% to your leaderboard {25:.1f}%")
    print(f"  2. If significantly different, discuss with your professor")
    print(f"  3. Try testing on multiple held-out patients for confirmation")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
