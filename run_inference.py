import config
from src.data_loader import load_all_holdout_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.feature_selection import select_features
from src.inference import make_inference, generate_submission_file
from src.utils import save_cache, load_cache
import os
import joblib

def run_inference():
    print(f"--- Sleep Scoring Inference - Iteration {config.CURRENT_ITERATION} ---")

    # Load the trained model (assuming it was saved during training)
    model_filename = f"model_iter{config.CURRENT_ITERATION}.joblib"
    model = load_cache(model_filename, config.CACHE_DIR)
    if model is None:
        print("Error: Trained model not found. Please run main.py first to train a model.")
        return

    # Load feature selector info (critical for Iteration 3+)
    selector_filename = f"selector_info_iter{config.CURRENT_ITERATION}.joblib"
    selector_info = load_cache(selector_filename, config.CACHE_DIR)
    if config.CURRENT_ITERATION >= 3 and selector_info is None:
        print(f"Warning: Feature selector not found for Iteration {config.CURRENT_ITERATION}.")
        print("Feature selection will be skipped, which may lead to incorrect predictions!")
        print("Please ensure you run main.py to train and save the feature selector.")
    elif selector_info is not None:
        print(f"✓ Loaded feature selector for Iteration {config.CURRENT_ITERATION}")

    # 1. Load Hold-out Data
    print("\n=== STEP 1: LOADING HOLDOUT DATA ===")
    holdout_data, all_epoch_ids, combined_record_ids, channel_info = load_all_holdout_data(config.HOLDOUT_DIR)

    print("\nLoaded holdout multi-channel data:")
    if 'eeg' in holdout_data:
        print(f"  EEG: {holdout_data['eeg'].shape}")
    if 'eog' in holdout_data:
        print(f"  EOG: {holdout_data['eog'].shape}")
    if 'emg' in holdout_data:
        print(f"  EMG: {holdout_data['emg'].shape}")
    print(f"  Total epochs: {len(all_epoch_ids)}")
    print(f"  Total records: {len(set(combined_record_ids))}")

    # 2. Preprocessing (using the same logic as training)
    print("\n=== STEP 2: PREPROCESSING ===")
    preprocessed_holdout_data = None
    cache_filename_preprocess_holdout = f"preprocessed_holdout_data_iter{config.CURRENT_ITERATION}.joblib"
    if config.USE_CACHE:
        preprocessed_holdout_data = load_cache(cache_filename_preprocess_holdout, config.CACHE_DIR)
        if preprocessed_holdout_data is not None:
            print("Loaded preprocessed holdout data from cache")

    if preprocessed_holdout_data is None:
        preprocessed_holdout_data = preprocess(holdout_data, channel_info, config, combined_record_ids)
        if config.USE_CACHE:
            save_cache(preprocessed_holdout_data, cache_filename_preprocess_holdout, config.CACHE_DIR)
            print("Saved preprocessed holdout data to cache")

    # 3. Feature Extraction (using the same logic as training)
    print("\n=== STEP 3: FEATURE EXTRACTION ===")
    holdout_features = None
    cache_filename_features_holdout = f"features_holdout_iter{config.CURRENT_ITERATION}.joblib"
    if config.USE_CACHE:
        holdout_features = load_cache(cache_filename_features_holdout, config.CACHE_DIR)
        if holdout_features is not None:
            print("Loaded holdout features from cache")

    if holdout_features is None:
        holdout_features = extract_features(preprocessed_holdout_data, config, channel_info)
        print(f"Extracted holdout features shape: {holdout_features.shape}")
        if config.USE_CACHE:
            save_cache(holdout_features, cache_filename_features_holdout, config.CACHE_DIR)
            print("Saved holdout features to cache")

    # 4. Feature Selection (CRITICAL: Apply same selection as training)
    print("\n=== STEP 4: FEATURE SELECTION ===")
    if selector_info is not None:
        print("Applying pre-fitted feature selection from training...")
        selected_holdout_features, _ = select_features(
            holdout_features,
            labels=None,  # No labels for holdout data
            config=config,
            selector_info=selector_info
        )
    else:
        print("No feature selection (using all features)")
        selected_holdout_features = holdout_features

    print(f"Final holdout features shape: {selected_holdout_features.shape}")

    # 5. Make Inference
    print("\n=== STEP 5: INFERENCE ===")
    predictions = make_inference(model, selected_holdout_features, config)

    # Dummy record and epoch numbers for submission file
    record_numbers = combined_record_ids # Assuming one record for dummy data
    epoch_numbers = all_epoch_ids

    # 6. Generate Submission File
    print("\n=== STEP 6: GENERATING SUBMISSION ===")
    generate_submission_file(predictions, record_numbers, epoch_numbers, config)

    print("\n" + "="*50)
    print("INFERENCE PIPELINE FINISHED")
    print("="*50)
    print(f"✓ Predictions generated for {len(predictions)} epochs")
    print(f"✓ Submission file saved")
    print("\n--- Inference Finished ---")

if __name__ == "__main__":
    run_inference()
