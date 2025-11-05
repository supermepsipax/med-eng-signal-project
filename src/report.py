import numpy as np
from datetime import datetime


def generate_report(model, features, labels, config, processing_log, metrics_dict=None):
    """
    Generates a report summarizing the results.
    
    Args:
        model (object): The trained model.
        features (np.ndarray): The input features.
        labels (np.ndarray): The corresponding labels.
        config (module): The configuration module.
        processing_log (str): Captured console output from pipeline.
        metrics_dict (dict): Dictionary containing all performance metrics.
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

