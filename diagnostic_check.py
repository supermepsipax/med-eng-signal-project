"""
Diagnostic script to identify SVM performance issues.

Run this to check:
1. Feature scaling issues
2. Class imbalance
3. Feature validity (NaN, Inf)
4. Prediction distribution

Usage:
    python diagnostic_check.py
"""

import numpy as np
import config
from src.data_loader import load_training_data
from src.preprocessing import preprocess_data
from src.feature_extraction import extract_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd


def check_feature_scaling(features):
    """Check if features have wildly different scales."""
    print("\n" + "="*70)
    print("FEATURE SCALING CHECK")
    print("="*70)

    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0)
    feature_mins = np.min(features, axis=0)
    feature_maxs = np.max(features, axis=0)
    feature_ranges = feature_maxs - feature_mins

    print(f"\nFeature statistics across {features.shape[1]} features:")
    print(f"  Mean range: [{feature_means.min():.2e}, {feature_means.max():.2e}]")
    print(f"  Std range:  [{feature_stds.min():.2e}, {feature_stds.max():.2e}]")
    print(f"  Min range:  [{feature_mins.min():.2e}, {feature_maxs.max():.2e}]")
    print(f"  Value range: [{feature_ranges.min():.2e}, {feature_ranges.max():.2e}]")

    # Check for huge scale differences
    scale_ratio = feature_ranges.max() / (feature_ranges.min() + 1e-10)
    print(f"\n⚠️  Scale ratio (max range / min range): {scale_ratio:.2e}")

    if scale_ratio > 100:
        print("❌ PROBLEM: Features have very different scales!")
        print("   → SVM performance will be poor without standardization")
        print("   → SOLUTION: Use StandardScaler before training")
    else:
        print("✓ Features are reasonably scaled")

    # Show examples of extreme features
    top_5_range = np.argsort(feature_ranges)[-5:]
    bottom_5_range = np.argsort(feature_ranges)[:5]

    print(f"\nTop 5 features by range:")
    for idx in top_5_range[::-1]:
        print(f"  Feature {idx}: range = {feature_ranges[idx]:.2e}, mean = {feature_means[idx]:.2e}")

    print(f"\nBottom 5 features by range:")
    for idx in bottom_5_range:
        print(f"  Feature {idx}: range = {feature_ranges[idx]:.2e}, mean = {feature_means[idx]:.2e}")


def check_class_imbalance(labels):
    """Check class distribution."""
    print("\n" + "="*70)
    print("CLASS IMBALANCE CHECK")
    print("="*70)

    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    print(f"\nClass distribution ({total} total epochs):")
    for stage_idx, count in zip(unique, counts):
        stage_name = stage_names[stage_idx]
        percentage = count / total * 100
        print(f"  {stage_name}: {count:5d} epochs ({percentage:5.1f}%)")

    # Check imbalance ratio
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count

    print(f"\n⚠️  Imbalance ratio (most common / least common): {imbalance_ratio:.1f}:1")

    if imbalance_ratio > 3:
        print("❌ PROBLEM: Significant class imbalance!")
        print("   → SVM will bias toward majority class (N2)")
        print("   → SOLUTION: Use class_weight='balanced' in SVC()")
    else:
        print("✓ Classes are reasonably balanced")


def check_feature_validity(features):
    """Check for NaN and Inf values."""
    print("\n" + "="*70)
    print("FEATURE VALIDITY CHECK")
    print("="*70)

    nan_count = np.isnan(features).sum()
    inf_count = np.isinf(features).sum()

    print(f"\nInvalid values in feature matrix:")
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")

    if nan_count > 0 or inf_count > 0:
        print("❌ PROBLEM: Features contain invalid values!")
        print("   → This will cause SVM to fail or give poor results")
        print("   → SOLUTION: Check feature extraction code for division by zero")

        # Find which features have issues
        if nan_count > 0:
            nan_features = np.where(np.isnan(features).any(axis=0))[0]
            print(f"\nFeatures with NaN: {nan_features[:10]}...")

        if inf_count > 0:
            inf_features = np.where(np.isinf(features).any(axis=0))[0]
            print(f"\nFeatures with Inf: {inf_features[:10]}...")
    else:
        print("✓ No NaN or Inf values found")


def compare_scaled_vs_unscaled(X_train, X_test, y_train, y_test):
    """Train SVM with and without scaling to show the difference."""
    print("\n" + "="*70)
    print("SCALING IMPACT TEST")
    print("="*70)

    # Test 1: SVM without scaling (current approach)
    print("\n1. SVM WITHOUT feature scaling:")
    svm_unscaled = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
    svm_unscaled.fit(X_train, y_train)
    y_pred_unscaled = svm_unscaled.predict(X_test)
    acc_unscaled = accuracy_score(y_test, y_pred_unscaled)
    print(f"   Accuracy: {acc_unscaled:.3f}")

    # Check prediction distribution
    pred_unique, pred_counts = np.unique(y_pred_unscaled, return_counts=True)
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    print("   Prediction distribution:")
    for stage_idx, count in zip(pred_unique, pred_counts):
        pct = count / len(y_pred_unscaled) * 100
        print(f"     {stage_names[stage_idx]}: {count} ({pct:.1f}%)")

    # Test 2: SVM with scaling (recommended)
    print("\n2. SVM WITH feature scaling (StandardScaler):")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_scaled = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
    svm_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = svm_scaled.predict(X_test_scaled)
    acc_scaled = accuracy_score(y_test, y_pred_scaled)
    print(f"   Accuracy: {acc_scaled:.3f}")

    # Check prediction distribution
    pred_unique, pred_counts = np.unique(y_pred_scaled, return_counts=True)
    print("   Prediction distribution:")
    for stage_idx, count in zip(pred_unique, pred_counts):
        pct = count / len(y_pred_scaled) * 100
        print(f"     {stage_names[stage_idx]}: {count} ({pct:.1f}%)")

    # Test 3: SVM with scaling AND class balancing (best practice)
    print("\n3. SVM WITH scaling AND class_weight='balanced':")
    svm_balanced = SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', random_state=42)
    svm_balanced.fit(X_train_scaled, y_train)
    y_pred_balanced = svm_balanced.predict(X_test_scaled)
    acc_balanced = accuracy_score(y_test, y_pred_balanced)
    print(f"   Accuracy: {acc_balanced:.3f}")

    # Check prediction distribution
    pred_unique, pred_counts = np.unique(y_pred_balanced, return_counts=True)
    print("   Prediction distribution:")
    for stage_idx, count in zip(pred_unique, pred_counts):
        pct = count / len(y_pred_balanced) * 100
        print(f"     {stage_names[stage_idx]}: {count} ({pct:.1f}%)")

    # Summary
    print("\n" + "-"*70)
    print("COMPARISON SUMMARY:")
    print("-"*70)
    print(f"  No scaling:                    {acc_unscaled:.3f} (your current approach)")
    print(f"  With scaling:                  {acc_scaled:.3f} (improvement: {acc_scaled - acc_unscaled:+.3f})")
    print(f"  With scaling + class balance:  {acc_balanced:.3f} (improvement: {acc_balanced - acc_unscaled:+.3f})")

    improvement_pct = (acc_balanced - acc_unscaled) / acc_unscaled * 100
    print(f"\n💡 Expected improvement: {improvement_pct:.1f}%")

    if improvement_pct > 10:
        print("🎯 Scaling and class balancing will significantly improve your SVM!")
    elif improvement_pct > 5:
        print("✓ Scaling and class balancing will moderately improve your SVM")
    else:
        print("⚠️  Small improvement - there may be other issues with features")


def main():
    print("="*70)
    print("SVM DIAGNOSTIC TOOL")
    print("="*70)
    print("\nThis script will identify common issues causing poor SVM performance.")
    print("Loading data and extracting features...")

    # Load and preprocess data
    data, labels = load_training_data(config)
    preprocessed_data = preprocess_data(data, config)
    features = extract_features(preprocessed_data, config)

    print(f"\n✓ Loaded {features.shape[0]} epochs with {features.shape[1]} features")

    # Run diagnostic checks
    check_feature_validity(features)
    check_feature_scaling(features)
    check_class_imbalance(labels)

    # Train/test split
    print("\n" + "="*70)
    print("SPLITTING DATA FOR COMPARISON TEST")
    print("="*70)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"  Training: {X_train.shape[0]} epochs")
    print(f"  Testing:  {X_test.shape[0]} epochs")

    # Compare scaled vs unscaled
    compare_scaled_vs_unscaled(X_train, X_test, y_train, y_test)

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("\n1. ✓ ALWAYS use StandardScaler for SVM:")
    print("   from sklearn.preprocessing import StandardScaler")
    print("   scaler = StandardScaler()")
    print("   X_train_scaled = scaler.fit_transform(X_train)")
    print("   X_test_scaled = scaler.transform(X_test)")
    print("\n2. ✓ Use class_weight='balanced' for imbalanced data:")
    print("   model = SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced')")
    print("\n3. ✓ Check for NaN/Inf in features before training")
    print("\n4. ✓ For final optimization, tune hyperparameters with GridSearchCV")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
