"""
Feature Blacklist for Domain-Shifted Features

Based on statistical analysis (analyze_feature_distributions.py), these features
showed the largest distribution differences between training and holdout datasets:
- Mean differences >50%
- Variance ratios >2x or <0.5x
- KS test failures (p < 0.05)

Blacklisting these features forces the model to rely only on robust, domain-invariant
features that should generalize better to the holdout data.
"""

# Features with >50% mean difference between training and holdout
DOMAIN_SHIFTED_FEATURES = [
    # Absolute statistical features (amplitude-dependent even after normalization)
    'eeg_ch0_mean',
    'eeg_ch0_median',
    'eeg_ch0_variance',
    'eeg_ch0_std',
    'eeg_ch0_skewness',
    'eeg_ch1_median',
    'eeg_ch1_variance',
    'eeg_ch1_std',

    # EOG variance features (showed 60%+ difference)
    'eog_ch0_eog_variance',
    'eog_ch1_eog_variance',

    # Additional features with large variance ratio differences
    'eeg_ch0_kurtosis',
    'eeg_ch1_kurtosis',
    'eeg_ch0_transition_ratio',  # 0.49x ratio

    # Zero-variance features after per-epoch normalization
    # These become constants (=1) after z-score normalization and carry no information
    'rms',              # sqrt(variance) = 1 after normalization
    'hjorth_activity',  # variance = 1 after normalization
    'eog_rms',          # same issue for EOG
    'emg_rms',          # same issue for EMG

    # Features with >2x variance ratio between datasets
    'theta_alpha_ratio',  # 6.31x ratio - unstable across domains
]

# Features that are more robust (ratios, normalized, relative)
# These should be PRIORITIZED for selection
ROBUST_FEATURES_PRIORITY = [
    # Power band ratios (domain-invariant)
    'delta_beta_ratio',
    # 'theta_alpha_ratio',  # REMOVED: 6.31x variance ratio - unstable across domains
    'theta_alertness_ratio',

    # Relative power (normalized to total power)
    'delta_rel',
    'theta_rel',
    'alpha_rel',
    'beta_rel',
    'gamma_rel',

    # Spectral shape features
    'spectral_edge_95',
    'spectral_entropy',
    'peak_frequency',

    # Temporal/shape features
    'zero_crossings',
    'hjorth_mobility',
    'hjorth_complexity',

    # EOG features (mostly ratios)
    'eog_rem_score',
    'eog_sem_score',
    'eog_sem_rem_ratio',
    'eog_burst_count',  # Iteration 4
    'eog_burst_density',  # Iteration 4

    # EMG features (relative measures)
    'emg_percentile_10',  # Iteration 4
    'emg_min_ratio',  # Iteration 4
    'emg_low_tone_proportion',  # Iteration 4
]


def is_blacklisted(feature_name):
    """
    Check if a feature should be excluded due to domain shift.

    Args:
        feature_name (str): Full feature name (e.g., 'eeg_ch0_mean')

    Returns:
        bool: True if feature should be excluded
    """
    # Check exact matches first
    if feature_name in DOMAIN_SHIFTED_FEATURES:
        return True

    # Check partial matches (handles channel variations)
    for blacklisted in DOMAIN_SHIFTED_FEATURES:
        # Remove channel suffix for comparison
        base_blacklisted = blacklisted.replace('_ch0', '').replace('_ch1', '')
        base_feature = feature_name.replace('_ch0', '').replace('_ch1', '')

        if base_feature.endswith(base_blacklisted.split('_', 1)[-1]):
            return True

    return False


def filter_blacklisted_features(features, feature_names):
    """
    Remove blacklisted features from feature matrix.

    Args:
        features (np.ndarray): Feature matrix (n_samples, n_features)
        feature_names (list): List of feature names

    Returns:
        tuple: (filtered_features, filtered_feature_names, removed_features)
    """
    import numpy as np

    keep_mask = np.array([not is_blacklisted(name) for name in feature_names])
    removed_features = [name for name in feature_names if is_blacklisted(name)]

    filtered_features = features[:, keep_mask]
    filtered_feature_names = [name for name in feature_names if not is_blacklisted(name)]

    return filtered_features, filtered_feature_names, removed_features


def get_robustness_score(feature_name):
    """
    Get a robustness score for prioritizing features during selection.

    Higher score = more robust to domain shift.

    Args:
        feature_name (str): Full feature name

    Returns:
        float: Robustness score (0-1, higher is better)
    """
    # Blacklisted features get score of 0
    if is_blacklisted(feature_name):
        return 0.0

    # Check if feature matches any priority patterns
    base_name = feature_name.split('_', 2)[-1] if '_ch' in feature_name else feature_name

    for robust_pattern in ROBUST_FEATURES_PRIORITY:
        if robust_pattern in base_name:
            return 1.0

    # Default score for non-blacklisted, non-priority features
    return 0.5
