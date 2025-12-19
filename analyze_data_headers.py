"""
EDF Header and Metadata Analysis

This script compares the EDF file headers and recording parameters between
training and holdout datasets to identify any differences in data collection,
sampling rates, channel configurations, or other recording parameters.

Usage:
    python analyze_data_headers.py
"""

import mne
import os
import numpy as np
from pathlib import Path
from glob import glob
import config

def analyze_edf_file(edf_path):
    """Extract detailed metadata from an EDF file."""
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

    info = {
        'filename': Path(edf_path).name,
        'duration_sec': raw.times[-1],
        'duration_hours': raw.times[-1] / 3600,
        'n_channels': len(raw.ch_names),
        'channel_names': raw.ch_names,
        'sampling_rates': {},
        'recording_date': raw.info.get('meas_date', 'Unknown'),
        'highpass': raw.info.get('highpass', 'Unknown'),
        'lowpass': raw.info.get('lowpass', 'Unknown'),
    }

    # Get sampling rate for each channel
    for ch_name in raw.ch_names:
        ch_idx = raw.ch_names.index(ch_name)
        info['sampling_rates'][ch_name] = raw.info['sfreq']

    # Identify channel types
    eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch.upper() or
                    any(p in ch.upper() for p in ['C3', 'C4', 'F3', 'F4', 'O1', 'O2', 'CZ', 'FZ', 'PZ'])]
    eog_channels = [ch for ch in raw.ch_names if 'EOG' in ch.upper()]
    emg_channels = [ch for ch in raw.ch_names if 'EMG' in ch.upper() or 'CHIN' in ch.upper()]
    other_channels = [ch for ch in raw.ch_names if ch not in eeg_channels + eog_channels + emg_channels]

    info['eeg_channels'] = eeg_channels
    info['eog_channels'] = eog_channels
    info['emg_channels'] = emg_channels
    info['other_channels'] = other_channels

    # Get signal statistics (sample a portion to avoid loading all data)
    raw_sample = raw.copy().crop(tmin=0, tmax=min(60, raw.times[-1]))
    data_sample = raw_sample.get_data()

    info['signal_stats'] = {
        'mean': np.mean(data_sample, axis=1),
        'std': np.std(data_sample, axis=1),
        'min': np.min(data_sample, axis=1),
        'max': np.max(data_sample, axis=1),
    }

    return info

def compare_channel_configs(train_infos, holdout_infos):
    """Compare channel configurations between datasets."""
    print("\n" + "="*80)
    print("CHANNEL CONFIGURATION COMPARISON")
    print("="*80)

    # Get common channel structure
    train_channels = [sorted(info['channel_names']) for info in train_infos]
    holdout_channels = [sorted(info['channel_names']) for info in holdout_infos]

    # Check if all training files have same channels
    train_consistent = all(ch == train_channels[0] for ch in train_channels)
    holdout_consistent = all(ch == holdout_channels[0] for ch in holdout_channels)

    print(f"\n📋 Channel Consistency:")
    print(f"  Training files: {'✓ All identical' if train_consistent else '✗ INCONSISTENT'}")
    print(f"  Holdout files:  {'✓ All identical' if holdout_consistent else '✗ INCONSISTENT'}")

    if train_consistent and holdout_consistent:
        train_set = set(train_channels[0])
        holdout_set = set(holdout_channels[0])

        if train_set == holdout_set:
            print(f"\n✓ Training and holdout have IDENTICAL channel configurations")
        else:
            print(f"\n⚠️  Training and holdout have DIFFERENT channels:")
            only_train = train_set - holdout_set
            only_holdout = holdout_set - train_set

            if only_train:
                print(f"  Channels only in training: {only_train}")
            if only_holdout:
                print(f"  Channels only in holdout: {only_holdout}")

    # Compare channel types
    print(f"\n📊 Channel Type Breakdown:")
    print(f"\n{'Dataset':<15} {'EEG':<8} {'EOG':<8} {'EMG':<8} {'Other':<8}")
    print("-" * 50)

    for info in train_infos[:3]:  # Show first 3 examples
        print(f"Train: {info['filename']:<8} {len(info['eeg_channels']):<8} "
              f"{len(info['eog_channels']):<8} {len(info['emg_channels']):<8} "
              f"{len(info['other_channels']):<8}")

    print()
    for info in holdout_infos[:3]:  # Show first 3 examples
        print(f"Hold:  {info['filename']:<8} {len(info['eeg_channels']):<8} "
              f"{len(info['eog_channels']):<8} {len(info['emg_channels']):<8} "
              f"{len(info['other_channels']):<8}")

def compare_sampling_rates(train_infos, holdout_infos):
    """Compare sampling rates between datasets."""
    print("\n" + "="*80)
    print("SAMPLING RATE COMPARISON")
    print("="*80)

    # Get unique sampling rates
    train_rates = set()
    holdout_rates = set()

    for info in train_infos:
        for rate in info['sampling_rates'].values():
            train_rates.add(rate)

    for info in holdout_infos:
        for rate in info['sampling_rates'].values():
            holdout_rates.add(rate)

    print(f"\nTraining sampling rates: {sorted(train_rates)}")
    print(f"Holdout sampling rates:  {sorted(holdout_rates)}")

    if train_rates == holdout_rates:
        print("✓ Sampling rates are IDENTICAL")
    else:
        print("⚠️  Sampling rates are DIFFERENT!")
        print(f"  Only in training: {train_rates - holdout_rates}")
        print(f"  Only in holdout:  {holdout_rates - train_rates}")

def compare_signal_statistics(train_infos, holdout_infos):
    """Compare raw signal statistics between datasets."""
    print("\n" + "="*80)
    print("SIGNAL STATISTICS COMPARISON (First 60s sample)")
    print("="*80)

    # Average statistics across all training files
    train_means = [info['signal_stats']['mean'] for info in train_infos]
    train_stds = [info['signal_stats']['std'] for info in train_infos]

    holdout_means = [info['signal_stats']['mean'] for info in holdout_infos]
    holdout_stds = [info['signal_stats']['std'] for info in holdout_infos]

    # Calculate global statistics
    train_mean_avg = np.mean([np.mean(m) for m in train_means])
    train_std_avg = np.mean([np.mean(s) for s in train_stds])

    holdout_mean_avg = np.mean([np.mean(m) for m in holdout_means])
    holdout_std_avg = np.mean([np.mean(s) for s in holdout_stds])

    print(f"\n📈 Raw Signal Statistics:")
    print(f"{'Dataset':<15} {'Avg Mean':<15} {'Avg Std':<15}")
    print("-" * 50)
    print(f"{'Training':<15} {train_mean_avg:<15.2e} {train_std_avg:<15.2e}")
    print(f"{'Holdout':<15} {holdout_mean_avg:<15.2e} {holdout_std_avg:<15.2e}")

    # Check for major differences
    mean_ratio = abs(train_mean_avg / holdout_mean_avg) if holdout_mean_avg != 0 else float('inf')
    std_ratio = abs(train_std_avg / holdout_std_avg) if holdout_std_avg != 0 else float('inf')

    print(f"\n{'Mean ratio (train/holdout):':<30} {mean_ratio:.2f}")
    print(f"{'Std ratio (train/holdout):':<30} {std_ratio:.2f}")

    if 0.8 < mean_ratio < 1.2 and 0.8 < std_ratio < 1.2:
        print("✓ Signal amplitudes are SIMILAR")
    else:
        print("⚠️  Signal amplitudes are DIFFERENT - possible scaling or unit difference!")

def compare_recording_durations(train_infos, holdout_infos):
    """Compare recording durations."""
    print("\n" + "="*80)
    print("RECORDING DURATION COMPARISON")
    print("="*80)

    train_durations = [info['duration_hours'] for info in train_infos]
    holdout_durations = [info['duration_hours'] for info in holdout_infos]

    print(f"\n⏱️  Duration Statistics:")
    print(f"{'Dataset':<15} {'Min (hrs)':<12} {'Max (hrs)':<12} {'Mean (hrs)':<12}")
    print("-" * 55)
    print(f"{'Training':<15} {min(train_durations):<12.2f} "
          f"{max(train_durations):<12.2f} {np.mean(train_durations):<12.2f}")
    print(f"{'Holdout':<15} {min(holdout_durations):<12.2f} "
          f"{max(holdout_durations):<12.2f} {np.mean(holdout_durations):<12.2f}")

def main():
    print("="*80)
    print("EDF HEADER AND METADATA ANALYSIS")
    print("="*80)

    # Load training files
    train_files = sorted(glob(os.path.join(config.TRAINING_DIR, '*.edf')))
    holdout_files = sorted(glob(os.path.join(config.HOLDOUT_DIR, '*.edf')))

    print(f"\nFound {len(train_files)} training files")
    print(f"Found {len(holdout_files)} holdout files")

    if not train_files:
        print(f"ERROR: No training files found in {config.TRAINING_DIR}")
        return

    if not holdout_files:
        print(f"ERROR: No holdout files found in {config.HOLDOUT_DIR}")
        return

    # Analyze all files
    print("\n📂 Analyzing training files...")
    train_infos = []
    for f in train_files:
        print(f"  Processing {Path(f).name}...")
        train_infos.append(analyze_edf_file(f))

    print("\n📂 Analyzing holdout files...")
    holdout_infos = []
    for f in holdout_files:
        print(f"  Processing {Path(f).name}...")
        holdout_infos.append(analyze_edf_file(f))

    # Run comparisons
    compare_channel_configs(train_infos, holdout_infos)
    compare_sampling_rates(train_infos, holdout_infos)
    compare_recording_durations(train_infos, holdout_infos)
    compare_signal_statistics(train_infos, holdout_infos)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\n🔍 Key Findings:")
    print("  1. Check if channel configurations match")
    print("  2. Verify sampling rates are identical")
    print("  3. Look for signal amplitude differences (scaling issues)")
    print("  4. Consider recording duration differences")
    print("\n💡 Next Step:")
    print("  Run: python analyze_feature_distributions.py")
    print("  This will compare extracted features statistically.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
