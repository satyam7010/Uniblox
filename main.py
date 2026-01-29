"""
Main Pipeline Script for Insurance Enrollment Prediction

This script orchestrates the complete ML pipeline including:
- Data loading and preprocessing
- Exploratory data analysis
- Model training and evaluation
- Hyperparameter tuning
- Model persistence

Usage:
    python main.py --data employee_data.csv --tune --save

Author: Data Science Team
Date: 2026-01-29
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing import DataProcessor
from src.model_training import ModelTrainer, print_model_comparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Insurance Enrollment Prediction ML Pipeline'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='employee_data.csv',
        help='Path to the employee data CSV file'
    )
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.2,
        help='Proportion of data to use for testing (default: 0.2)'
    )
    parser.add_argument(
        '--tune', 
        action='store_true',
        help='Perform hyperparameter tuning on the best model'
    )
    parser.add_argument(
        '--save', 
        action='store_true',
        help='Save the trained model and preprocessor'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='models',
        help='Directory to save models (default: models)'
    )
    parser.add_argument(
        '--random-state', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()


def print_data_summary(stats: dict) -> None:
    """Print a summary of the data exploration results."""
    print("\n" + "=" * 80)
    print("DATA EXPLORATION SUMMARY")
    print("=" * 80)
    
    print(f"\nDataset Shape: {stats['shape'][0]} rows x {stats['shape'][1]} columns")
    
    # Missing values
    missing = {k: v for k, v in stats['missing_values'].items() if v > 0}
    if missing:
        print(f"\nMissing Values: {missing}")
    else:
        print("\nMissing Values: None")
    
    # Target distribution
    print("\nTarget Distribution (enrolled):")
    print(f"  Not Enrolled (0): {stats['target_distribution'].get(0, 0)} ({stats['target_percentage'].get(0, 0):.1%})")
    print(f"  Enrolled (1): {stats['target_distribution'].get(1, 0)} ({stats['target_percentage'].get(1, 0):.1%})")
    
    # Enrollment by segments
    print("\nEnrollment Rate by Employment Type:")
    for emp_type, rate in sorted(stats['enrollment_by_employment_type'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {emp_type}: {rate:.1%}")
    
    print("\nEnrollment Rate by Region:")
    for region, rate in sorted(stats['enrollment_by_region'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {region}: {rate:.1%}")
    
    print("\nEnrollment Rate by Has Dependents:")
    for dep, rate in stats['enrollment_by_dependents'].items():
        print(f"  {dep}: {rate:.1%}")
    
    # Numerical statistics
    print("\nNumerical Features Summary:")
    for col, col_stats in stats['numerical_stats'].items():
        print(f"\n  {col}:")
        print(f"    Mean: {col_stats['mean']:.2f}, Std: {col_stats['std']:.2f}")
        print(f"    Min: {col_stats['min']:.2f}, Max: {col_stats['max']:.2f}")
    
    print("=" * 80)


def save_results(
    results: dict, 
    trainer: ModelTrainer, 
    processor: DataProcessor,
    output_dir: str
) -> None:
    """Save training results and models to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training results
    results_file = os.path.join(output_dir, 'training_results.json')
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    serializable_results['best_model'] = trainer.best_model_name
    serializable_results['timestamp'] = datetime.now().isoformat()
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Results saved to {results_file}")
    
    # Save the best model
    model_file = os.path.join(output_dir, 'best_model.joblib')
    trainer.save_model(trainer.best_model, model_file)
    
    # Save the data processor (for preprocessing new data)
    import joblib
    processor_file = os.path.join(output_dir, 'data_processor.joblib')
    joblib.dump(processor, processor_file)
    logger.info(f"Data processor saved to {processor_file}")


def main():
    """Main function to run the complete ML pipeline."""
    args = parse_arguments()
    
    print("\n" + "=" * 80)
    print("INSURANCE ENROLLMENT PREDICTION PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data file: {args.data}")
    print(f"  Test size: {args.test_size}")
    print(f"  Hyperparameter tuning: {args.tune}")
    print(f"  Save models: {args.save}")
    print(f"  Random state: {args.random_state}")
    
    # Step 1: Data Processing
    logger.info("Step 1: Data Processing")
    processor = DataProcessor()
    X_train, X_test, y_train, y_test, stats = processor.prepare_data(
        args.data, 
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Print data summary
    print_data_summary(stats)
    
    print(f"\nProcessed Data Shape:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    
    # Step 2: Model Training and Comparison
    logger.info("Step 2: Model Training and Comparison")
    trainer = ModelTrainer(random_state=args.random_state)
    results = trainer.compare_models(X_train, y_train, X_test, y_test)
    
    # Print model comparison results
    print_model_comparison(results)
    
    # Step 3: Hyperparameter Tuning (optional)
    if args.tune:
        logger.info("Step 3: Hyperparameter Tuning")
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING")
        print("=" * 80)
        
        # Tune the best model
        best_model, best_params = trainer.hyperparameter_tuning(
            trainer.best_model_name,
            X_train,
            y_train
        )
        
        print(f"\nBest Parameters for {trainer.best_model_name}:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # Re-evaluate the tuned model
        tuned_metrics = trainer.evaluate_model(best_model, X_test, y_test)
        print(f"\nTuned Model Performance:")
        print(f"  Accuracy: {tuned_metrics['accuracy']:.4f}")
        print(f"  Precision: {tuned_metrics['precision']:.4f}")
        print(f"  Recall: {tuned_metrics['recall']:.4f}")
        print(f"  F1 Score: {tuned_metrics['f1']:.4f}")
        print(f"  ROC-AUC: {tuned_metrics.get('roc_auc', 'N/A')}")
        
        # Update the best model
        trainer.best_model = best_model
        results[trainer.best_model_name]['tuned_results'] = tuned_metrics
    
    # Step 4: Feature Importance
    logger.info("Step 4: Feature Importance Analysis")
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    
    importance_df = trainer.get_feature_importance(
        trainer.best_model, 
        X_train.columns.tolist()
    )
    
    if importance_df is not None:
        print("\nTop 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Step 5: Save Results and Models (optional)
    if args.save:
        logger.info("Step 5: Saving Results and Models")
        save_results(results, trainer, processor, args.output_dir)
        print(f"\nModels and results saved to '{args.output_dir}/' directory")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nBest Model: {trainer.best_model_name}")
    
    best_test = results[trainer.best_model_name]['test_results']
    print(f"Final Test Performance:")
    print(f"  Accuracy: {best_test['accuracy']:.4f}")
    print(f"  Precision: {best_test['precision']:.4f}")
    print(f"  Recall: {best_test['recall']:.4f}")
    print(f"  F1 Score: {best_test['f1']:.4f}")
    if 'roc_auc' in best_test:
        print(f"  ROC-AUC: {best_test['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = best_test['confusion_matrix']
    print(f"  [[TN={cm[0][0]}, FP={cm[0][1]}]")
    print(f"   [FN={cm[1][0]}, TP={cm[1][1]}]]")
    
    return trainer, processor, results


if __name__ == "__main__":
    trainer, processor, results = main()
