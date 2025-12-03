# Iteration 4 Implementation - Changes Summary

**Date**: 2025-12-03
**Status**: In Progress
**Purpose**: Document all code changes made to enhance Iteration 4 performance

---

## Overview

Iteration 4 focuses on optimizing the feature set and model performance for sleep stage classification. Building on Iteration 3's multi-signal processing foundation, Iteration 4 adds REM-specific features to improve discrimination between similar sleep stages, particularly REM vs N1 and REM vs Wake.

### Key Changes
- Enhanced EOG features with REM burst detection (temporal pattern analysis)
- Enhanced EMG features with muscle atonia markers (REM-specific low tone detection)
- Feature name tracking and reporting system (enables verification of selected features)
- Feature count: 72 to 78 total features before selection

---

## File-by-File Changes

### 1. `src/feature_extraction.py` - REM-Specific Feature Engineering

#### Change 1.1: Enhanced EOG Features with REM Burst Detection
**Location**: `extract_eog_features()` function
**Type**: Feature Addition

**Rationale**:
REM sleep has a distinctive temporal pattern of bursts of rapid eye movements separated by quiescent periods. Previous features counted total REM movements but did not capture this burst structure, which distinguishes REM from other stages with eye movements.

**New Features (3 added)**:

1. **`eog_burst_count`**: Number of REM bursts in the epoch
   - Detects clusters of rapid eye movements separated by quiet periods
   - REM typically has 3-10 bursts per 30-second epoch
   - N1/Wake have fewer, more scattered movements
   - Algorithm: Find REM events, cluster by temporal proximity

2. **`eog_burst_density`**: Proportion of epoch containing active REM bursts
   - Ratio of time in active bursts to total epoch duration
   - REM: 0.3-0.6 (30-60% of epoch has bursts)
   - Other stages: <0.2
   - Normalized metric (0-1 range)

3. **`eog_mean_burst_duration`**: Average duration of REM bursts
   - Mean length of continuous burst periods
   - REM bursts typically last 1-3 seconds
   - Captures sustained vs sporadic eye movements
   - Measured in seconds

**Implementation Details**:
- Uses existing REM-filtered signal (>1 Hz component)
- Identifies REM events using peak detection
- Clusters events within 0.5-second windows to define bursts
- Calculates burst statistics from clustered events

**Impact**:
- EOG features per channel: 8 to 11 (+3 burst features)
- Total EOG features (2 channels): 16 to 22 (+6 features)
- Expected improvement in REM vs N1 discrimination

---

#### Change 1.2: Enhanced EMG Features with Muscle Atonia Markers
**Location**: `extract_emg_features()` function
**Type**: Feature Addition

**Rationale**:
REM sleep is characterized by muscle atonia, the lowest muscle tone of any sleep stage. Previous features (RMS, std, mean absolute value) measured overall muscle activity but did not specifically capture sustained low tone. These new features target the minimum tone levels characteristic of REM.

**New Features (3 added)**:

1. **`emg_percentile_10`**: 10th percentile of signal amplitude
   - Captures sustained low muscle tone (not just brief dips)
   - REM: Very low values due to muscle atonia
   - Wake/NREM: Higher baseline tone
   - Robust to occasional muscle twitches in REM

2. **`emg_min_ratio`**: Ratio of minimum to mean amplitude
   - min(abs(signal)) / mean(abs(signal))
   - REM: Low ratio (minimum much lower than mean)
   - Wake: High ratio (minimum closer to mean)
   - Normalized metric for cross-subject comparison

3. **`emg_low_tone_proportion`**: Proportion of epoch below low-tone threshold
   - Percentage of samples below 0.3 × mean amplitude
   - REM: High proportion (sustained atonia)
   - Other stages: Lower proportion
   - Captures temporal persistence of low tone

**Implementation Details**:
- Uses absolute value of EMG signal for amplitude analysis
- Percentile calculated using numpy.percentile()
- Threshold for low tone: 0.3 × mean (empirically determined)
- All features normalized/ratio-based for robustness

**Impact**:
- EMG features: 4 to 7 (+3 atonia features)
- Expected improvement in REM vs Wake and REM vs NREM discrimination
- Better capture of muscle atonia physiology

---

#### Change 1.3: Feature Name Tracking and Reporting System
**Location**: Multiple files (`src/feature_extraction.py`, `src/feature_selection.py`, `src/report.py`, `main.py`, `run_inference.py`)
**Type**: Infrastructure Enhancement

**Rationale**:
Previously, features were tracked only as numpy arrays without names, making it difficult to verify which specific features were selected during feature selection. This change implements comprehensive feature tracking to enable verification of which features (especially the new Iteration 4 REM-specific features) are being used by the model.

**Changes Made**:

1. **Feature Extraction (`src/feature_extraction.py`)**:
   - Modified `extract_features()` to return tuple: `(features, feature_names)`
   - Feature names format: `{signal_type}_{channel_name}_{feature_name}`
   - Examples: `eeg_ch0_mean`, `eog_ch1_sem_score`, `emg_ch0_percentile_10`
   - Uses actual channel names from `channel_info` when available

2. **Feature Selection (`src/feature_selection.py`)**:
   - Added `feature_names` parameter to `select_features()`
   - Returns tuple: `(selected_features, selector_info, selection_report)`
   - Tracks correlation removal with paired feature names and correlation values
   - Tracks mutual information scores for all features with rankings
   - Generates comprehensive `selection_report` dictionary containing:
     - List of removed features during correlation analysis (grouped by signal type)
     - Correlation pairs: which features were removed and which were kept
     - All MI scores ranked by importance
     - Final selected features (grouped by signal type)
     - Special tracking for Iteration 4 REM-specific features

3. **Report Generation (`src/report.py`)**:
   - Added `selection_report` parameter to `generate_report()`
   - New "FEATURE SELECTION DETAILS" section in `report.txt` showing:
     - **Stage 1 (Correlation Analysis)**: Lists removed features by signal type with correlation pairs
     - **Stage 2 (Mutual Information Selection)**:
       - Top 20 features ranked by MI score (checkmark indicates selected)
       - Summary count by signal type (EEG/EOG/EMG)
       - Dedicated section for Iteration 4 REM-specific features with MI scores and ranks

4. **Training Pipeline (`main.py`)**:
   - Updated to handle new tuple returns from `extract_features()` and `select_features()`
   - Caches feature names alongside features
   - Saves selection report for documentation
   - Passes selection report to `generate_report()`

5. **Inference Pipeline (`run_inference.py`)**:
   - Updated to handle new tuple returns
   - Caches and loads feature names for holdout data
   - Maintains consistency with training feature names

**Impact**:
- Complete visibility into which features are selected/removed
- Verification that Iteration 4 REM-specific features are being used
- Ability to track feature importance via MI scores
- Detailed documentation in `report.txt` for analysis
- No functional changes to model training or inference, only enhanced tracking

**Example Report Output**:
```
======================================================================
FEATURE SELECTION DETAILS
======================================================================
Input Features: 78
Selected Features: 40

Stage 1: Correlation Analysis (threshold r > 0.95)
----------------------------------------------------------------------
Removed 12 highly correlated features:

  EEG features removed (8):
    - eeg_ch0_std (r=0.982 with eeg_ch0_variance)
    ...

Stage 2: Mutual Information Selection
----------------------------------------------------------------------
Selected top 40 features based on mutual information:

Rank   MI Score     Feature Name
----------------------------------------------------------------------
1      0.234567 ✓ eeg_ch0_delta_rel
2      0.198432 ✓ eeg_ch1_theta_rel
...

Iteration 4 REM-Specific Features Selected:
----------------------------------------------------------------------
  eog_ch0_burst_count                (MI: 0.156789, Rank: 8)
  emg_ch0_percentile_10              (MI: 0.138901, Rank: 14)
  ...
```

---

### 2. Feature Count Updates

**Updated Feature Totals**:

| Signal | Iteration 3 | Iteration 4 | Change |
|--------|-------------|-------------|--------|
| EEG (2 channels) | 2 × 26 = 52 | 2 × 26 = 52 | No change |
| EOG (2 channels) | 2 × 8 = 16 | 2 × 11 = 22 | +6 burst features |
| EMG (1 channel) | 1 × 4 = 4 | 1 × 7 = 7 | +3 atonia features |
| **Total** | **72** | **78** | **+6 features** |

After feature selection: 78 to 40 features (correlation + mutual information selection)

---

## Expected Performance Impact

### Iteration 4 vs Iteration 3

**Target Improvements**:
- REM sensitivity: +5-10% (better detection of REM episodes)
- REM precision: +5-10% (fewer false REM predictions)
- Overall F1 Macro: +2-5% (primarily from better REM classification)

**Physiological Basis**:
- Burst detection captures temporal structure of REM (not just frequency content)
- Atonia markers specifically target lowest muscle tone in REM
- Both features are highly specific to REM physiology

**Expected Confusion Matrix Changes**:
- Fewer REM misclassified as Wake (burst patterns differ)
- Fewer REM misclassified as N1 (atonia + burst pattern combined)
- Potentially fewer N1 misclassified as REM (N1 has SEM but no bursts and no atonia)

---

## Testing Instructions

**Clear cache** (feature extraction changed):
```bash
rm cache/*.joblib
```

**Retrain with new features**:
```bash
python main.py
```

**Check feature extraction output**:
- Should show "78 total features" before selection
- Should show "2 EOG channels × 11 features"
- Should show "1 EMG channel × 7 features"

**Verify new features in model**:
- Check feature selection output to see if new features are retained
- Review feature importance rankings (Random Forest feature_importances_)

**Expected results**:
- Improved REM F1 score (currently likely 65-75%, target 70-80%)
- Better overall F1 Macro (target 42-55% on leaderboard)
- More balanced confusion matrix for REM

---

## Technical Notes

### Rationale for Feature Design

**EOG Burst Detection**:
- REM has phasic (intermittent bursts) vs tonic (continuous) eye movements
- This temporal pattern is as important as frequency content
- Previous features: "how many rapid movements?" (frequency)
- New features: "how are movements clustered in time?" (temporal structure)
- Addresses the clinical definition of "bursts of rapid eye movements"

**EMG Atonia Markers**:
- REM muscle atonia is not just low activity, but sustained low tone
- Mean/RMS can be misleading (affected by brief twitches)
- Percentiles capture sustained minimum tone (robust to twitches)
- Low-tone proportion captures temporal persistence
- Necessary for REM vs Wake discrimination (both can have eye movements, but only REM has atonia)

### Feature Interactions

The new features work synergistically:
- High EOG burst count + Low EMG percentile = Strong REM indicator
- High SEM score + Low burst count = N1 (not REM)
- High EOG activity + High EMG tone = Wake (not REM)

This multi-signal integration provides discriminative power.

---

**Document Version**: 1.1
**Implementation Status**: Complete - Ready for Iteration 4 Testing
**Changes Implemented**:
- REM-specific EOG burst detection features (3 features)
- REM-specific EMG atonia marker features (3 features)
- Feature name tracking and reporting system (5 files modified)

**Next Steps**:
1. Clear cache: `rm cache/*.joblib`
2. Train model: `python main.py`
3. Verify feature counts in output (should show 78 features)
4. Check `report.txt` for feature selection details
5. Verify REM-specific features are selected (check MI scores and rankings)
6. Check REM performance in confusion matrix
7. Submit to leaderboard and compare with Iteration 3 baseline
