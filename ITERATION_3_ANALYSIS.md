# Iteration 3: Multi-Signal Processing - Code Analysis and Required Changes

**Date**: 2025-12-03
**Purpose**: Analyze current implementation against Iteration 3 requirements and document necessary changes

---

## Executive Summary

The codebase is **mostly ready** for Iteration 3 (EEG + EOG + EMG), but requires targeted modifications to EOG feature extraction and feature selection implementation.

### Current Status
- ✅ **Preprocessing**: Fully supports multi-channel processing (EEG + EOG + EMG)
- ⚠️ **Feature Extraction**: EEG and EMG are correct, but **EOG extracts 24 features instead of required 6**
- ❌ **Feature Selection**: Not implemented (placeholder only)
- ✅ **Classification**: Random Forest ready for Iteration 3
- ✅ **Inference**: Compatible with multi-signal pipeline

---

## Iteration 3 Requirements (from PROJECT_GUIDE.md)

### Target Feature Counts (lines 1897-1921)

| Signal Type | Channels | Features per Channel | Total Features | Purpose |
|-------------|----------|---------------------|----------------|---------|
| **EEG** | 2 | 24 (14 time + 10 freq) | 48 | Primary sleep staging signal |
| **EOG** | 2 | **6** (eye movement specific) | **12** | REM detection, eye movements |
| **EMG** | 1 | 2-4 (muscle tone) | 2-4 | Muscle atonia detection (REM) |
| **TOTAL** | 5 | - | **62-64 features** | Before selection |
| **AFTER SELECTION** | - | - | **30-50 features** | Selected subset |

### Key Points
1. **EOG features should be minimal and focused** on eye movement detection
2. Feature selection becomes critical to reduce from 62-64 to 30-50 features
3. Random Forest classifier with hyperparameter tuning

---

## Detailed File Analysis

### 1. ✅ `src/preprocessing.py` - READY

**Status**: Fully supports Iteration 3

**Current Implementation**:
- `preprocess_multi_channel()` handles EEG, EOG, EMG separately (lines 128-196)
- Iteration 3 check: `if config.CURRENT_ITERATION >= 3` (line 169)
- Applies appropriate filtering per signal type:
  - EEG: Bandpass filter (0.5-40 Hz typical)
  - EOG: Lowpass filter (20 Hz cutoff for 50 Hz sampling)
  - EMG: Lowpass filter (60 Hz cutoff for 125 Hz sampling)

**Recommendation**: ✅ No changes needed

---

### 2. ⚠️ `src/feature_extraction.py` - NEEDS MODIFICATION

**Status**: EOG feature extraction needs reduction from 24 to 6 features

#### Problem: `extract_eog_features()` (lines 325-359)

**Current Implementation**:
```python
def extract_eog_features(eog_signal, fs=50, include_frequency=True):
    # Uses same 14 time-domain features as EEG
    features = extract_time_domain_features(eog_signal)

    if include_frequency:
        # Adds 10 frequency-domain features
        freq_features = extract_frequency_domain_features(
            eog_signal, fs, signal_type="eog"
        )
        features.update(freq_features)

    # Result: 24 features (14 time + 10 freq)
```

**Issue**: Extracts **24 features** but PROJECT_GUIDE.md specifies **only 6 features** (line 1902-1905):
- Peak amplitude (max absolute value)
- Variance
- REM detection score (rapid deflections counting)
- 3 additional eye movement specific features

#### Required Changes to `extract_eog_features()`

Replace the current implementation with eye-movement-specific features:

```python
def extract_eog_features(eog_signal, fs=50, include_frequency=False):
    """
    Extract 6 EOG-specific features focused on eye movement detection.

    EOG signals are used to detect:
    - Rapid eye movements (REM sleep indicator)
    - Slow eye movements (NREM sleep)
    - Eye blinks and artifacts

    PROJECT_GUIDE.md Iteration 3 specification (lines 1902-1905):
    - ~6 features per channel (vs 24 for EEG)
    - Focused on REM detection and eye movements

    Args:
        eog_signal (np.ndarray): 1D array of EOG signal data
        fs (float): Sampling frequency (default 50 Hz for EOG)
        include_frequency (bool): Kept for API compatibility, but EOG uses simplified features

    Returns:
        dict: 6 EOG-specific features focused on eye movement detection
    """
    # 1. Peak amplitude (max absolute value) - eye movement magnitude
    peak_amplitude = np.max(np.abs(eog_signal))

    # 2. Variance - signal variability indicator
    variance = np.var(eog_signal)

    # 3. RMS - signal power (normalized measure)
    rms = np.sqrt(np.mean(eog_signal**2))

    # 4. REM detection score - count rapid deflections
    # High-pass filter >0.5 Hz to isolate rapid movements
    from scipy.signal import butter, filtfilt
    nyquist = fs / 2
    if 0.5 < nyquist:  # Only apply if possible
        b, a = butter(3, 0.5 / nyquist, btype='high')
        filtered_signal = filtfilt(b, a, eog_signal)
    else:
        filtered_signal = eog_signal

    # Count peaks above threshold (indicates rapid eye movements)
    threshold = 0.5 * np.std(filtered_signal)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(np.abs(filtered_signal), height=threshold)
    rem_score = len(peaks)  # More peaks = likely REM

    # 5. Zero-crossing rate - frequency of signal changes
    zero_crossings = np.sum(np.diff(np.sign(eog_signal)) != 0)

    # 6. Mean absolute value - overall activity level
    mean_abs_value = np.mean(np.abs(eog_signal))

    features = {
        "eog_peak_amplitude": peak_amplitude,
        "eog_variance": variance,
        "eog_rms": rms,
        "eog_rem_score": rem_score,
        "eog_zero_crossings": zero_crossings,
        "eog_mean_abs_value": mean_abs_value,
    }

    return features
```

**Impact**:
- Reduces EOG features from 24 → 6 per channel
- Total features: 2 EEG × 24 + 2 EOG × 6 + 1 EMG × 3 = **63 features** (matches guide)

---

### 3. ✅ `src/feature_extraction.py` - EMG (Minor Enhancement)

**Current Implementation** (`extract_emg_features()`, lines 362-382):
```python
def extract_emg_features(emg_signal):
    features = {
        "emg_mean": np.mean(emg_signal),
        "emg_std": np.std(emg_signal),
        "emg_rms": np.sqrt(np.mean(emg_signal**2)),
    }
    # TODO: Add high-frequency power ratio
```

**Status**: Extracts 3 features (target: 2-4) ✅

**Optional Enhancement** (add high-frequency power for better REM detection):

```python
def extract_emg_features(emg_signal, fs=125):
    """
    Extract EMG-specific features for muscle tone detection.

    EMG signals are used to detect:
    - Muscle tone levels (high in wake, low in REM due to atonia)
    - Muscle twitches and artifacts

    PROJECT_GUIDE.md Iteration 3 specification (lines 1907-1914):
    - 2-4 features per channel
    - Signal power (mean squared amplitude)
    - Variance
    - Optional: High-frequency (20-40 Hz) power ratio

    Args:
        emg_signal (np.ndarray): 1D array of EMG signal data
        fs (float): Sampling frequency (default 125 Hz)

    Returns:
        dict: 4 EMG-specific features for muscle tone quantification
    """
    # 1. RMS - signal power (low in REM, high in wake/NREM)
    emg_rms = np.sqrt(np.mean(emg_signal**2))

    # 2. Standard deviation - signal variability
    emg_std = np.std(emg_signal)

    # 3. Mean absolute value - overall activity level
    emg_mean_abs = np.mean(np.abs(emg_signal))

    # 4. High-frequency power ratio (20-40 Hz) - muscle activity indicator
    from scipy.signal import welch
    freqs, psd = welch(emg_signal, fs=fs, nperseg=min(256, len(emg_signal)))

    # Calculate power in high-frequency band (20-40 Hz)
    hf_idx = np.logical_and(freqs >= 20, freqs <= 40)
    total_idx = freqs <= 60  # Total band

    hf_power = np.trapz(psd[hf_idx], freqs[hf_idx]) if np.sum(hf_idx) > 0 else 0
    total_power = np.trapz(psd[total_idx], freqs[total_idx]) if np.sum(total_idx) > 0 else 1e-10

    hf_ratio = hf_power / total_power

    features = {
        "emg_rms": emg_rms,
        "emg_std": emg_std,
        "emg_mean_abs": emg_mean_abs,
        "emg_hf_ratio": hf_ratio,
    }

    return features
```

**Impact**: EMG features: 3 → 4 (still within 2-4 target range)

---

### 4. ❌ `src/feature_selection.py` - NEEDS IMPLEMENTATION

**Status**: Placeholder only, not functional

**Current Implementation** (lines 36-52):
```python
if config.CURRENT_ITERATION <= 2:
    # Early iterations: Use all available features
    selected_features = features
elif config.CURRENT_ITERATION == 3:
    # TODO: Students should implement feature selection here
    print("TODO: Students should implement feature selection for iteration 3")
    selected_features = features  # No selection implemented yet
```

**Problem**: Does not reduce 63 features → 30-50 as required

#### Required Implementation

**Strategy** (from PROJECT_GUIDE.md lines 1768-1803):
1. **Stage 1**: Variance thresholding (remove low-variance features)
2. **Stage 2**: Correlation analysis (remove highly correlated features)
3. **Stage 3**: Statistical testing (select top-k using mutual information)

**Implementation**:

```python
def select_features(features, labels, config):
    """
    Multi-stage feature selection for Iteration 3+.

    Strategy (PROJECT_GUIDE.md lines 1768-1803):
    1. Variance Thresholding: Remove features with very low variance
    2. Correlation Analysis: Remove highly correlated features (r > 0.95)
    3. Statistical Testing: Select top-k using mutual information

    Target: Reduce from ~63 features to 30-50 features
    """
    print(f"Selecting features for iteration {config.CURRENT_ITERATION}...")
    print(f"Input features shape: {features.shape}")

    if features.shape[1] == 0:
        print("⚠️  WARNING: No features to select from!")
        return features

    if config.CURRENT_ITERATION <= 2:
        # Early iterations: Use all available features
        print("Early iteration - using all available features")
        selected_features = features

    elif config.CURRENT_ITERATION >= 3:
        print("\n" + "="*70)
        print("FEATURE SELECTION - MULTI-STAGE APPROACH")
        print("="*70)

        from sklearn.feature_selection import (
            VarianceThreshold,
            SelectKBest,
            mutual_info_classif
        )

        # Stage 1: Variance Thresholding
        print("\nStage 1: Variance Thresholding")
        print(f"  Input: {features.shape[1]} features")

        # Remove features with variance < 1% of max variance
        feature_variances = np.var(features, axis=0)
        threshold = 0.01 * np.max(feature_variances)

        variance_selector = VarianceThreshold(threshold=threshold)
        features_stage1 = variance_selector.fit_transform(features)
        print(f"  Removed {features.shape[1] - features_stage1.shape[1]} low-variance features")
        print(f"  Output: {features_stage1.shape[1]} features")

        # Stage 2: Correlation Analysis
        print("\nStage 2: Correlation Analysis")
        print(f"  Input: {features_stage1.shape[1]} features")

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(features_stage1, rowvar=False)

        # Find highly correlated feature pairs (r > 0.95)
        high_corr_pairs = np.where(np.abs(corr_matrix) > 0.95)
        high_corr_pairs = [(i, j) for i, j in zip(*high_corr_pairs) if i < j]

        # Remove one feature from each correlated pair
        features_to_remove = set()
        for i, j in high_corr_pairs:
            if i not in features_to_remove:
                features_to_remove.add(j)

        features_to_keep = [i for i in range(features_stage1.shape[1])
                           if i not in features_to_remove]
        features_stage2 = features_stage1[:, features_to_keep]

        print(f"  Removed {len(features_to_remove)} highly correlated features (r > 0.95)")
        print(f"  Output: {features_stage2.shape[1]} features")

        # Stage 3: Mutual Information Selection
        print("\nStage 3: Mutual Information Selection")
        print(f"  Input: {features_stage2.shape[1]} features")

        # Select top-k features using mutual information
        target_k = getattr(config, 'FEATURE_SELECTION_K', 40)
        k = min(target_k, features_stage2.shape[1])

        mi_selector = SelectKBest(mutual_info_classif, k=k)
        selected_features = mi_selector.fit_transform(features_stage2, labels)

        print(f"  Selected top {k} features based on mutual information")
        print(f"  Output: {selected_features.shape[1]} features")

        print("\n" + "="*70)
        print(f"FINAL: Reduced from {features.shape[1]} to {selected_features.shape[1]} features")
        print("="*70)

    return selected_features
```

**Configuration Addition** (add to `config.py`):
```python
# Feature Selection (Iteration 3+)
FEATURE_SELECTION_K = 40  # Target number of features after selection (30-50 range)
```

**Impact**: Properly reduces features from ~63 → 40 as required

---

### 5. ✅ `src/classification.py` - READY

**Status**: Random Forest implementation ready for Iteration 3

**Current Implementation**:
- `train_classifier()` routes to `train_random_forest()` for iteration ≥3 (line 64)
- `train_random_forest()` uses GridSearchCV for hyperparameter tuning (lines 351-419)
- Hyperparameters: n_estimators, max_depth, min_samples_split, min_samples_leaf

**Recommendation**: ✅ No changes needed

---

### 6. ✅ `src/inference.py` - READY

**Status**: Compatible with multi-signal pipeline

**Current Implementation**:
- Handles scaler if present (for SVM compatibility)
- Prints feature statistics for debugging
- Warns about distribution mismatches

**Recommendation**: ✅ No changes needed

---

## Summary of Required Changes

### High Priority (Required for Iteration 3)

1. **`src/feature_extraction.py`**: ⚠️ **MUST MODIFY**
   - **Line 325-359**: Rewrite `extract_eog_features()` to return 6 features instead of 24
   - Focus on eye movement detection (peak amplitude, variance, REM score, zero-crossings, mean absolute value)
   - **Impact**: Total features: 48 (EEG) + 12 (EOG) + 3 (EMG) = **63 features**

2. **`src/feature_selection.py`**: ❌ **MUST IMPLEMENT**
   - **Line 36-61**: Implement multi-stage feature selection for iteration 3
   - Reduce from ~63 features to 30-50 using variance threshold, correlation removal, and mutual information
   - **Impact**: Final feature count: **~40 features** (configurable)

### Medium Priority (Optional Enhancement)

3. **`src/feature_extraction.py`**: ✅ **OPTIONAL**
   - **Line 362-382**: Enhance `extract_emg_features()` to include high-frequency power ratio (20-40 Hz)
   - Already has 3 features (meets 2-4 requirement), 4th adds more discriminative power for REM
   - **Impact**: EMG features: 3 → 4

### No Changes Needed

4. **`src/preprocessing.py`**: ✅ Already supports multi-channel
5. **`src/classification.py`**: ✅ Random Forest ready
6. **`src/inference.py`**: ✅ Compatible with pipeline

---

## Expected Feature Counts by Iteration

| Iteration | Signals | Features Before Selection | Features After Selection | Classifier |
|-----------|---------|---------------------------|-------------------------|------------|
| **1** | EEG (2 ch) | 2 × 14 = 28 | 28 (no selection) | k-NN |
| **2** | EEG (2 ch) + EOG (2 ch) | 2×24 + 2×24 = 96 | 96 (no selection) | SVM |
| **3** | EEG (2 ch) + EOG (2 ch) + EMG (1 ch) | 2×24 + 2×**6** + 1×4 = **64** | **30-50** (selection) | Random Forest |
| **4** | All signals | 64 | Optimized subset | RF (tuned) |

---

## Testing Checklist

After implementing changes, verify:

- [ ] EOG features: 6 per channel × 2 channels = 12 total
- [ ] EMG features: 4 per channel × 1 channel = 4 total
- [ ] EEG features: 24 per channel × 2 channels = 48 total
- [ ] **Total before selection**: 64 features
- [ ] **Total after selection**: 30-50 features (check `selected_features.shape`)
- [ ] Feature selection runs without errors
- [ ] Random Forest training completes
- [ ] Model can make predictions on holdout data
- [ ] No NaN or Inf values in features

---

## Implementation Priority

### Immediate (Must Do)
1. Modify `extract_eog_features()` to return 6 features
2. Implement feature selection in `select_features()` for iteration 3

### Optional (Nice to Have)
3. Add high-frequency ratio to `extract_emg_features()`

### Verification
4. Test full pipeline with iteration 3 configuration
5. Check feature counts at each stage
6. Verify model training and prediction works

---

## References

- **PROJECT_GUIDE.md**: Lines 1870-1923 (Iteration 3 specification)
- **PROJECT_GUIDE.md**: Lines 1768-1803 (Feature selection strategy)
- **PROJECT_GUIDE.md**: Lines 1897-1921 (Target feature counts)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-03
