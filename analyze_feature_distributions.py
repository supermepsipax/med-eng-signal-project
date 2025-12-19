"""
Feature Distribution Analysis

This script performs statistical analysis on extracted features from both
training and holdout datasets to identify distribution shifts, feature
mismatches, or domain adaptation issues.

Usage:
    python analyze_feature_distributions.py
"""

import config
from src.data_loader import load_all_training_data, load_all_holdout_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.utils import load_cache
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def compute_feature_statistics(features, feature_names, dataset_name):
    """Compute comprehensive statistics for features."""
    stats_dict = {
        'dataset': dataset_name,
        'n_samples': features.shape[0],
        'n_features': features.shape[1],
        'feature_names': feature_names,
        'mean': np.mean(features, axis=0),
        'std': np.std(features, axis=0),
        'min': np.min(features, axis=0),
        'max': np.max(features, axis=0),
        'median': np.median(features, axis=0),
        'q25': np.percentile(features, 25, axis=0),
        'q75': np.percentile(features, 75, axis=0),
        'nan_count': np.sum(np.isnan(features), axis=0),
        'inf_count': np.sum(np.isinf(features), axis=0),
    }

    # Identify problematic features
    stats_dict['zero_variance_features'] = np.where(stats_dict['std'] < 1e-10)[0]
    stats_dict['high_nan_features'] = np.where(stats_dict['nan_count'] > features.shape[0] * 0.1)[0]
    stats_dict['extreme_value_features'] = np.where(np.abs(stats_dict['max']) > 1e10)[0]

    return stats_dict

def compare_feature_distributions(train_stats, holdout_stats):
    """Compare feature distributions between training and holdout."""
    print("\n" + "="*80)
    print("FEATURE DISTRIBUTION COMPARISON")
    print("="*80)

    print(f"\n📊 Dataset Sizes:")
    print(f"  Training: {train_stats['n_samples']} samples, {train_stats['n_features']} features")
    print(f"  Holdout:  {holdout_stats['n_samples']} samples, {holdout_stats['n_features']} features")

    if train_stats['n_features'] != holdout_stats['n_features']:
        print("\n⚠️  CRITICAL: Feature count MISMATCH!")
        print("  Training and holdout have different number of features!")
        print("  This will cause prediction failures.")
        return

    n_features = train_stats['n_features']

    # Compare feature statistics
    print(f"\n📈 Statistical Comparison:")

    # Mean absolute difference
    mean_diff = np.abs(train_stats['mean'] - holdout_stats['mean'])
    mean_diff_pct = np.where(np.abs(train_stats['mean']) > 1e-10,
                             100 * mean_diff / np.abs(train_stats['mean']),
                             0)

    # Standard deviation ratio
    std_ratio = np.where(train_stats['std'] > 1e-10,
                        holdout_stats['std'] / train_stats['std'],
                        1.0)

    print(f"\n{'Metric':<30} {'Min':<12} {'Max':<12} {'Median':<12}")
    print("-" * 70)
    print(f"{'Mean difference (abs)':<30} {np.min(mean_diff):<12.2e} "
          f"{np.max(mean_diff):<12.2e} {np.median(mean_diff):<12.2e}")
    print(f"{'Mean difference (%)':<30} {np.min(mean_diff_pct):<12.1f} "
          f"{np.max(mean_diff_pct):<12.1f} {np.median(mean_diff_pct):<12.1f}")
    print(f"{'Std ratio (holdout/train)':<30} {np.min(std_ratio):<12.2f} "
          f"{np.max(std_ratio):<12.2f} {np.median(std_ratio):<12.2f}")

    # Identify features with large distribution shifts
    large_shift_threshold = 50  # 50% difference
    large_shift_features = np.where(mean_diff_pct > large_shift_threshold)[0]

    print(f"\n⚠️  Features with >50% mean difference: {len(large_shift_features)}")
    if len(large_shift_features) > 0:
        print(f"  Showing first 10 of {len(large_shift_features)}:")
        for i in large_shift_features[:10]:
            feat_name = train_stats['feature_names'][i] if i < len(train_stats['feature_names']) else f"Feature_{i}"
            print(f"    {feat_name}: {mean_diff_pct[i]:.1f}% difference")

    # Identify features with very different variance
    variance_mismatch = np.where((std_ratio < 0.5) | (std_ratio > 2.0))[0]
    print(f"\n⚠️  Features with 2x variance difference: {len(variance_mismatch)}")
    if len(variance_mismatch) > 0:
        print(f"  Showing first 10 of {len(variance_mismatch)}:")
        for i in variance_mismatch[:10]:
            feat_name = train_stats['feature_names'][i] if i < len(train_stats['feature_names']) else f"Feature_{i}"
            print(f"    {feat_name}: {std_ratio[i]:.2f}x ratio")

    return mean_diff_pct, std_ratio

def check_data_quality(train_stats, holdout_stats):
    """Check for data quality issues."""
    print("\n" + "="*80)
    print("DATA QUALITY CHECK")
    print("="*80)

    print(f"\n🔍 Training Data Quality:")
    print(f"  Zero-variance features: {len(train_stats['zero_variance_features'])}")
    print(f"  High NaN features (>10%): {len(train_stats['high_nan_features'])}")
    print(f"  Extreme value features (>1e10): {len(train_stats['extreme_value_features'])}")

    print(f"\n🔍 Holdout Data Quality:")
    print(f"  Zero-variance features: {len(holdout_stats['zero_variance_features'])}")
    print(f"  High NaN features (>10%): {len(holdout_stats['high_nan_features'])}")
    print(f"  Extreme value features (>1e10): {len(holdout_stats['extreme_value_features'])}")

    # Check for features that are problematic in one but not the other
    train_zero_var = set(train_stats['zero_variance_features'])
    holdout_zero_var = set(holdout_stats['zero_variance_features'])

    only_train_zero = train_zero_var - holdout_zero_var
    only_holdout_zero = holdout_zero_var - train_zero_var

    if only_holdout_zero:
        print(f"\n⚠️  {len(only_holdout_zero)} features have zero variance in HOLDOUT but not TRAINING!")
        print("  This suggests holdout data has less diversity or different recording conditions.")
        if len(only_holdout_zero) <= 10:
            print(f"  Features: {only_holdout_zero}")

    if only_train_zero:
        print(f"\n⚠️  {len(only_train_zero)} features have zero variance in TRAINING but not HOLDOUT!")
        print("  This is unusual and may indicate feature extraction issues.")

def perform_statistical_tests(train_features, holdout_features, feature_names, n_samples=10):
    """Perform statistical tests on a sample of features."""
    print("\n" + "="*80)
    print("STATISTICAL HYPOTHESIS TESTS")
    print("="*80)

    print(f"\nPerforming Kolmogorov-Smirnov tests on {n_samples} random features...")
    print("(Tests if training and holdout come from the same distribution)\n")

    # Sample random features
    n_features = min(train_features.shape[1], n_samples)
    feature_indices = np.random.choice(train_features.shape[1], n_features, replace=False)

    print(f"{'Feature Name':<40} {'KS Statistic':<15} {'p-value':<15} {'Same Dist?'}")
    print("-" * 85)

    different_dist_count = 0
    for idx in feature_indices:
        feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"

        # Remove NaN/Inf values
        train_vals = train_features[:, idx]
        holdout_vals = holdout_features[:, idx]

        train_vals = train_vals[np.isfinite(train_vals)]
        holdout_vals = holdout_vals[np.isfinite(holdout_vals)]

        if len(train_vals) > 0 and len(holdout_vals) > 0:
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(train_vals, holdout_vals)

            # p-value < 0.05 suggests distributions are different
            same_dist = "✓ Yes" if p_value > 0.05 else "✗ NO"
            if p_value <= 0.05:
                different_dist_count += 1

            print(f"{feat_name:<40} {ks_stat:<15.4f} {p_value:<15.4e} {same_dist}")

    print(f"\n📊 Summary: {different_dist_count}/{n_features} tested features have significantly different distributions")
    print(f"  (p < 0.05 indicates distributions are different)")

    if different_dist_count > n_features * 0.5:
        print("\n⚠️  CRITICAL: >50% of tested features have different distributions!")
        print("  This suggests a significant domain shift between training and holdout data.")

def visualize_feature_comparison(train_features, holdout_features, feature_names,
                                 mean_diff_pct, std_ratio, output_file='feature_comparison.png'):
    """Create visualization comparing training and holdout features."""
    print(f"\n📊 Creating visualization: {output_file}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Mean difference distribution
    ax1 = axes[0, 0]
    valid_diffs = mean_diff_pct[np.isfinite(mean_diff_pct)]
    ax1.hist(valid_diffs, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Mean Difference (%)')
    ax1.set_ylabel('Number of Features')
    ax1.set_title('Distribution of Mean Differences\n(Training vs Holdout)')
    ax1.axvline(50, color='red', linestyle='--', label='50% threshold')
    ax1.legend()
    ax1.set_xlim([0, min(200, np.percentile(valid_diffs, 99))])

    # Plot 2: Std ratio distribution
    ax2 = axes[0, 1]
    valid_ratios = std_ratio[np.isfinite(std_ratio)]
    ax2.hist(valid_ratios, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Std Ratio (Holdout/Training)')
    ax2.set_ylabel('Number of Features')
    ax2.set_title('Distribution of Variance Ratios')
    ax2.axvline(0.5, color='red', linestyle='--', label='2x thresholds')
    ax2.axvline(2.0, color='red', linestyle='--')
    ax2.legend()
    ax2.set_xlim([0, min(3, np.percentile(valid_ratios, 99))])

    # Plot 3: Feature means comparison (scatter)
    ax3 = axes[1, 0]
    train_means = np.mean(train_features, axis=0)
    holdout_means = np.mean(holdout_features, axis=0)

    # Use log scale if values span many orders of magnitude
    valid_mask = np.isfinite(train_means) & np.isfinite(holdout_means)
    ax3.scatter(train_means[valid_mask], holdout_means[valid_mask], alpha=0.3, s=10)
    ax3.set_xlabel('Training Mean')
    ax3.set_ylabel('Holdout Mean')
    ax3.set_title('Feature Means: Training vs Holdout')

    # Add diagonal line
    min_val = min(np.min(train_means[valid_mask]), np.min(holdout_means[valid_mask]))
    max_val = max(np.max(train_means[valid_mask]), np.max(holdout_means[valid_mask]))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect match')
    ax3.legend()

    # Plot 4: Feature stds comparison (scatter)
    ax4 = axes[1, 1]
    train_stds = np.std(train_features, axis=0)
    holdout_stds = np.std(holdout_features, axis=0)

    valid_mask = np.isfinite(train_stds) & np.isfinite(holdout_stds)
    ax4.scatter(train_stds[valid_mask], holdout_stds[valid_mask], alpha=0.3, s=10)
    ax4.set_xlabel('Training Std')
    ax4.set_ylabel('Holdout Std')
    ax4.set_title('Feature Stds: Training vs Holdout')

    # Add diagonal line
    min_val = min(np.min(train_stds[valid_mask]), np.min(holdout_stds[valid_mask]))
    max_val = max(np.max(train_stds[valid_mask]), np.max(holdout_stds[valid_mask]))
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect match')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {output_file}")

def main():
    print("="*80)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*80)

    # Step 1: Load and extract training features
    print("\n=== STEP 1: LOADING TRAINING DATA ===")
    multi_channel_data, labels, record_ids, channel_info = load_all_training_data(
        config.TRAINING_DIR,
        epoch_length=30
    )

    print("\n=== STEP 2: PREPROCESSING TRAINING DATA ===")
    preprocessed_train = preprocess(multi_channel_data, channel_info, config, record_ids)

    print("\n=== STEP 3: EXTRACTING TRAINING FEATURES ===")
    train_features, train_feature_names = extract_features(preprocessed_train, config, channel_info)
    print(f"Training features shape: {train_features.shape}")

    # Step 2: Load and extract holdout features
    print("\n=== STEP 4: LOADING HOLDOUT DATA ===")
    holdout_data, all_epoch_ids, holdout_record_ids, holdout_channel_info = load_all_holdout_data(
        config.HOLDOUT_DIR
    )

    print("\n=== STEP 5: PREPROCESSING HOLDOUT DATA ===")
    preprocessed_holdout = preprocess(holdout_data, holdout_channel_info, config, holdout_record_ids)

    print("\n=== STEP 6: EXTRACTING HOLDOUT FEATURES ===")
    holdout_features, holdout_feature_names = extract_features(preprocessed_holdout, config, holdout_channel_info)
    print(f"Holdout features shape: {holdout_features.shape}")

    # Step 3: Compute statistics
    print("\n=== STEP 7: COMPUTING STATISTICS ===")
    train_stats = compute_feature_statistics(train_features, train_feature_names, "Training")
    holdout_stats = compute_feature_statistics(holdout_features, holdout_feature_names, "Holdout")

    # Step 4: Compare distributions
    mean_diff_pct, std_ratio = compare_feature_distributions(train_stats, holdout_stats)

    # Step 5: Check data quality
    check_data_quality(train_stats, holdout_stats)

    # Step 6: Statistical tests
    perform_statistical_tests(train_features, holdout_features, train_feature_names, n_samples=20)

    # Step 7: Visualize
    visualize_feature_comparison(train_features, holdout_features, train_feature_names,
                                mean_diff_pct, std_ratio)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*80)

    print("\n🔍 Analysis Complete. Key questions to answer:")
    print("  1. Are feature counts identical? (If not, pipeline is broken)")
    print("  2. Do features have similar means/stds? (Large differences = domain shift)")
    print("  3. Are there features with very different distributions? (Check KS tests)")
    print("  4. Are there data quality issues? (NaN, Inf, zero variance)")

    print("\n💡 If you found significant differences:")
    print("  ► This explains why leaderboard scores (~25-28%) differ from your")
    print("    training/validation scores (~60%+)")
    print("  ► Your model trained on one domain, but leaderboard tests on another")
    print("  ► Consider domain adaptation techniques or discuss with professor")

    print("\n📊 Visualization saved to: feature_comparison.png")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
