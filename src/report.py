import numpy as np
from datetime import datetime


def generate_report(model, features, labels, config, processing_log, metrics_dict=None, selection_report=None):
    """
    Generates a report summarizing the results.

    Args:
        model (object): The trained model.
        features (np.ndarray): The input features.
        labels (np.ndarray): The corresponding labels.
        config (module): The configuration module.
        processing_log (str): Captured console output from pipeline.
        metrics_dict (dict): Dictionary containing all performance metrics.
        selection_report (dict): Detailed feature selection tracking information (optional).
    """
    print("Generating report...")
    
    if metrics_dict is None:
        print("⚠️  Warning: No metrics available for report generation")
        return
    
    # Get metrics
    cv_results = metrics_dict.get('cv_results', {})
    cm = metrics_dict.get('confusion_matrix')
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    
    # Build report
    report_content = f"""
{'='*70}
SLEEP SCORING REPORT - ITERATION {config.CURRENT_ITERATION}
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}
MODEL INFORMATION
{'='*70}
Model Type: {type(model).__name__}
Classifier: {metrics_dict.get('classifier_type', 'Unknown').upper()}
Number of Features: {metrics_dict.get('n_features', 0)}
Number of Samples: {metrics_dict.get('n_samples', 0)} epochs

{'='*70}
PERFORMANCE METRICS
{'='*70}
Accuracy:      {cv_results.get('mean_accuracy', 0):.4f} (± {cv_results.get('std_accuracy', 0):.4f})
Kappa:         {cv_results.get('mean_kappa', 0):.4f} (± {cv_results.get('std_kappa', 0):.4f})
F1-Score:      {cv_results.get('mean_f1', 0):.4f} (± {cv_results.get('std_f1', 0):.4f})

{'='*70}
CONFUSION MATRIX
{'='*70}
Predicted →
Actual ↓    {'  '.join(f'{name:>6}' for name in stage_names)}
"""
    
    # Add confusion matrix
    if cm is not None:
        for i, stage_name in enumerate(stage_names):
            row_values = '  '.join(f'{cm[i][j]:>6}' for j in range(len(stage_names)))
            report_content += f"{stage_name:<8}    {row_values}\n"

    # Add feature selection details if available
    if selection_report is not None:
        report_content += f"""
{'='*70}
FEATURE SELECTION DETAILS
{'='*70}
Input Features: {selection_report['n_input_features']}
Selected Features: {selection_report['n_selected_features']}

"""

        # Correlation removal details
        if selection_report['correlation_removed']:
            report_content += f"""Stage 1: Correlation Analysis (threshold r > 0.95)
{'-'*70}
Removed {len(selection_report['correlation_removed'])} highly correlated features:

"""
            # Group removed features by signal type
            eeg_removed = [f for f in selection_report['correlation_removed'] if f.startswith('eeg_')]
            eog_removed = [f for f in selection_report['correlation_removed'] if f.startswith('eog_')]
            emg_removed = [f for f in selection_report['correlation_removed'] if f.startswith('emg_')]

            if eeg_removed:
                report_content += f"  EEG features removed ({len(eeg_removed)}):\n"
                for pair in selection_report['correlation_kept_pairs']:
                    if pair['removed'].startswith('eeg_'):
                        report_content += f"    - {pair['removed']} (r={pair['correlation']:.3f} with {pair['kept']})\n"

            if eog_removed:
                report_content += f"\n  EOG features removed ({len(eog_removed)}):\n"
                for pair in selection_report['correlation_kept_pairs']:
                    if pair['removed'].startswith('eog_'):
                        report_content += f"    - {pair['removed']} (r={pair['correlation']:.3f} with {pair['kept']})\n"

            if emg_removed:
                report_content += f"\n  EMG features removed ({len(emg_removed)}):\n"
                for pair in selection_report['correlation_kept_pairs']:
                    if pair['removed'].startswith('emg_'):
                        report_content += f"    - {pair['removed']} (r={pair['correlation']:.3f} with {pair['kept']})\n"
        else:
            report_content += f"""Stage 1: Correlation Analysis
{'-'*70}
No highly correlated features found (r > 0.95).

"""

        # Mutual information selection details
        if selection_report['mi_scores']:
            report_content += f"""
Stage 2: Mutual Information Selection
{'-'*70}
Selected top {selection_report['n_selected_features']} features based on mutual information:

"""
            # Show top 20 features by MI score
            report_content += f"{'Rank':<6} {'MI Score':<12} {'Feature Name'}\n"
            report_content += f"{'-'*70}\n"
            for item in selection_report['mi_scores'][:20]:
                selected_marker = "✓" if item['feature'] in selection_report['selected_features'] else " "
                report_content += f"{item['rank']:<6} {item['mi_score']:<12.6f} {selected_marker} {item['feature']}\n"

            if len(selection_report['mi_scores']) > 20:
                report_content += f"\n... ({len(selection_report['mi_scores']) - 20} more features)\n"

            # Summary by signal type
            report_content += f"\n\nSelected Features by Signal Type:\n{'-'*70}\n"
            eeg_selected = [f for f in selection_report['selected_features'] if f.startswith('eeg_')]
            eog_selected = [f for f in selection_report['selected_features'] if f.startswith('eog_')]
            emg_selected = [f for f in selection_report['selected_features'] if f.startswith('emg_')]

            report_content += f"  EEG: {len(eeg_selected)} features\n"
            report_content += f"  EOG: {len(eog_selected)} features\n"
            report_content += f"  EMG: {len(emg_selected)} features\n"

            # Show iteration 4 REM-specific features if they were selected
            rem_specific_features = [
                f for f in selection_report['selected_features']
                if any(marker in f for marker in ['burst_count', 'burst_density', 'mean_burst_duration',
                                                    'percentile_10', 'min_ratio', 'low_tone_proportion'])
            ]
            if rem_specific_features:
                report_content += f"\n\nIteration 4 REM-Specific Features Selected:\n{'-'*70}\n"
                for feat in rem_specific_features:
                    # Find the MI score for this feature
                    mi_item = next((item for item in selection_report['mi_scores'] if item['feature'] == feat), None)
                    if mi_item:
                        report_content += f"  {feat:<50} (MI: {mi_item['mi_score']:.6f}, Rank: {mi_item['rank']})\n"

    report_content += f"""
{'='*70}
DETAILED PROCESSING LOG
{'='*70}

{processing_log}

{'='*70}
END OF REPORT
{'='*70}
"""
    
    # Write report
    with open("report.txt", "w") as f:
        f.write(report_content)
    
    print("Report saved to report.txt")

