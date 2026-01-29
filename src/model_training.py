"""
Model Training Module for Insurance Enrollment Prediction

This module handles model training, hyperparameter tuning, and model persistence.
Implements multiple classification algorithms with cross-validation.

Author: Data Science Team
Date: 2026-01-29
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import logging
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    A class to handle model training, evaluation, and selection for insurance enrollment prediction.
    
    Attributes:
        models (dict): Dictionary of model instances
        best_model: The best performing model after comparison
        best_model_name (str): Name of the best model
        training_results (dict): Results from training all models
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ModelTrainer with default models.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = self._initialize_models()
        self.best_model = None
        self.best_model_name = None
        self.training_results = {}
        
    def _initialize_models(self) -> Dict[str, Any]:
        """
        Initialize a dictionary of classification models with default parameters.
        
        Returns:
            Dict containing model instances
        """
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5
            ),
            'svm': SVC(
                random_state=self.random_state,
                probability=True,
                class_weight='balanced'
            )
        }
        return models
    
    def cross_validate_models(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        cv: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation on all models.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict containing cross-validation results for each model
        """
        logger.info(f"Performing {cv}-fold cross-validation on {len(self.models)} models")
        
        cv_results = {}
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            logger.info(f"Cross-validating {name}...")
            
            # Calculate multiple scoring metrics
            accuracy_scores = cross_val_score(model, X, y, cv=cv_splitter, scoring='accuracy')
            f1_scores = cross_val_score(model, X, y, cv=cv_splitter, scoring='f1')
            roc_auc_scores = cross_val_score(model, X, y, cv=cv_splitter, scoring='roc_auc')
            
            cv_results[name] = {
                'accuracy_mean': accuracy_scores.mean(),
                'accuracy_std': accuracy_scores.std(),
                'f1_mean': f1_scores.mean(),
                'f1_std': f1_scores.std(),
                'roc_auc_mean': roc_auc_scores.mean(),
                'roc_auc_std': roc_auc_scores.std()
            }
            
            logger.info(f"  {name} - Accuracy: {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std():.4f})")
        
        return cv_results
    
    def train_model(
        self, 
        model_name: str, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Any:
        """
        Train a specific model on the training data.
        
        Args:
            model_name (str): Name of the model to train
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Trained model instance
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        logger.info(f"Training {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        logger.info(f"{model_name} training complete")
        
        return model
    
    def train_all_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Train all models on the training data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Dict containing all trained models
        """
        logger.info("Training all models")
        
        trained_models = {}
        for name in self.models.keys():
            trained_models[name] = self.train_model(name, X_train, y_train)
        
        return trained_models
    
    def evaluate_model(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model instance
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict containing evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        return metrics
    
    def compare_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train and compare all models, selecting the best one.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict containing results for all models
        """
        logger.info("Comparing all models")
        
        results = {}
        best_score = 0
        
        # First, perform cross-validation
        cv_results = self.cross_validate_models(X_train, y_train)
        
        # Train and evaluate each model
        for name, model in self.models.items():
            logger.info(f"Training and evaluating {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics = self.evaluate_model(model, X_test, y_test)
            
            # Combine CV and test results
            results[name] = {
                'cv_results': cv_results[name],
                'test_results': metrics
            }
            
            # Track best model (using F1 score as primary metric)
            if metrics['f1'] > best_score:
                best_score = metrics['f1']
                self.best_model = model
                self.best_model_name = name
        
        self.training_results = results
        logger.info(f"Best model: {self.best_model_name} with F1 score: {best_score:.4f}")
        
        return results
    
    def hyperparameter_tuning(
        self, 
        model_name: str, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        param_grid: Dict[str, List] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model_name (str): Name of the model to tune
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            param_grid (Dict): Parameter grid for GridSearchCV
            
        Returns:
            Tuple containing the best estimator and best parameters
        """
        logger.info(f"Performing hyperparameter tuning for {model_name}")
        
        # Default parameter grids
        default_param_grids = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'min_samples_split': [2, 5, 10]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        }
        
        if param_grid is None:
            param_grid = default_param_grids.get(model_name, {})
        
        if not param_grid:
            logger.warning(f"No parameter grid defined for {model_name}")
            return self.models[model_name], {}
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update the model with best estimator
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def get_feature_importance(
        self, 
        model: Any, 
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Args:
            model: Trained model instance
            feature_names (List[str]): List of feature names
            
        Returns:
            pd.DataFrame: Feature importance dataframe sorted by importance
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            logger.warning("Model does not support feature importance extraction")
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model: Any, filepath: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model instance
            filepath (str): Path to save the model
        """
        logger.info(f"Saving model to {filepath}")
        joblib.dump(model, filepath)
        logger.info("Model saved successfully")
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Loaded model instance
        """
        logger.info(f"Loading model from {filepath}")
        model = joblib.load(filepath)
        logger.info("Model loaded successfully")
        return model


def print_model_comparison(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Pretty print model comparison results.
    
    Args:
        results (Dict): Results from compare_models method
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    
    # Create summary table
    summary = []
    for model_name, model_results in results.items():
        cv = model_results['cv_results']
        test = model_results['test_results']
        summary.append({
            'Model': model_name,
            'CV Accuracy': f"{cv['accuracy_mean']:.4f} (+/- {cv['accuracy_std']:.4f})",
            'CV F1': f"{cv['f1_mean']:.4f} (+/- {cv['f1_std']:.4f})",
            'Test Accuracy': f"{test['accuracy']:.4f}",
            'Test F1': f"{test['f1']:.4f}",
            'Test ROC-AUC': f"{test.get('roc_auc', 'N/A'):.4f}" if 'roc_auc' in test else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    # Test the model trainer
    from data_processing import DataProcessor
    
    # Prepare data
    processor = DataProcessor()
    X_train, X_test, y_train, y_test, stats = processor.prepare_data("employee_data.csv")
    
    # Train and compare models
    trainer = ModelTrainer()
    results = trainer.compare_models(X_train, y_train, X_test, y_test)
    
    # Print results
    print_model_comparison(results)
    
    # Print feature importance for best model
    importance_df = trainer.get_feature_importance(trainer.best_model, X_train.columns.tolist())
    if importance_df is not None:
        print("\n=== Feature Importance (Top 10) ===")
        print(importance_df.head(10).to_string(index=False))
