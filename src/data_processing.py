"""
Data Processing Module for Insurance Enrollment Prediction

This module handles all data loading, cleaning, and preprocessing operations
including feature engineering and data splitting.

Author: Data Science Team
Date: 2026-01-29
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    A class to handle all data processing operations for the insurance enrollment dataset.
    
    Attributes:
        scaler (StandardScaler): Scaler for numerical features
        label_encoders (dict): Dictionary of label encoders for categorical features
        feature_columns (list): List of feature column names used for training
    """
    
    def __init__(self):
        """Initialize the DataProcessor with empty scalers and encoders."""
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns = []
        self.categorical_columns = ['gender', 'marital_status', 'employment_type', 'region', 'has_dependents']
        self.numerical_columns = ['age', 'salary', 'tenure_years']
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load the employee data from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def explore_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform exploratory data analysis on the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict containing various statistics about the data
        """
        logger.info("Performing exploratory data analysis")
        
        stats = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'target_distribution': df['enrolled'].value_counts().to_dict(),
            'target_percentage': df['enrolled'].value_counts(normalize=True).to_dict(),
            'numerical_stats': df[self.numerical_columns].describe().to_dict(),
            'categorical_distributions': {
                col: df[col].value_counts().to_dict() 
                for col in self.categorical_columns
            }
        }
        
        # Calculate enrollment rate by different segments
        stats['enrollment_by_gender'] = df.groupby('gender')['enrolled'].mean().to_dict()
        stats['enrollment_by_employment_type'] = df.groupby('employment_type')['enrolled'].mean().to_dict()
        stats['enrollment_by_region'] = df.groupby('region')['enrolled'].mean().to_dict()
        stats['enrollment_by_dependents'] = df.groupby('has_dependents')['enrolled'].mean().to_dict()
        
        return stats
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and removing duplicates.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Cleaning data")
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Drop employee_id as it's not a feature
        if 'employee_id' in df_clean.columns:
            df_clean = df_clean.drop('employee_id', axis=1)
        
        # Check for missing values
        missing_count = df_clean.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values")
            # Fill numerical columns with median
            for col in self.numerical_columns:
                if df_clean[col].isnull().any():
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            # Fill categorical columns with mode
            for col in self.categorical_columns:
                if df_clean[col].isnull().any():
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        logger.info(f"Data cleaning complete. Final shape: {df_clean.shape}")
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing columns to improve model performance.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        logger.info("Engineering features")
        
        df_fe = df.copy()
        
        # Age-based features
        df_fe['age_group'] = pd.cut(
            df_fe['age'], 
            bins=[0, 30, 40, 50, 60, 100], 
            labels=['young', 'early_mid', 'mid', 'late_mid', 'senior']
        )
        
        # Salary-based features
        df_fe['salary_quartile'] = pd.qcut(
            df_fe['salary'], 
            q=4, 
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # Tenure-based features
        df_fe['is_new_employee'] = (df_fe['tenure_years'] < 1).astype(int)
        df_fe['is_long_tenure'] = (df_fe['tenure_years'] > 5).astype(int)
        
        # Interaction features
        df_fe['salary_per_tenure'] = df_fe['salary'] / (df_fe['tenure_years'] + 1)
        df_fe['age_salary_ratio'] = df_fe['age'] / df_fe['salary'] * 10000
        
        # Update categorical columns list with new features
        self.categorical_columns.extend(['age_group', 'salary_quartile'])
        
        logger.info(f"Created {5} new features")
        return df_fe
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit the encoders (True for training, False for inference)
            
        Returns:
            pd.DataFrame: Dataframe with encoded features
        """
        logger.info("Encoding categorical features")
        
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col in df_encoded.columns:
                if fit:
                    # Fit and transform
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    # Transform only
                    if col in self.label_encoders:
                        # Handle unseen labels
                        df_encoded[col] = df_encoded[col].astype(str)
                        df_encoded[col] = df_encoded[col].apply(
                            lambda x: self.label_encoders[col].transform([x])[0] 
                            if x in self.label_encoders[col].classes_ 
                            else -1
                        )
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        logger.info("Scaling numerical features")
        
        df_scaled = df.copy()
        
        # Extended numerical columns with engineered features
        num_cols = self.numerical_columns + ['salary_per_tenure', 'age_salary_ratio']
        num_cols = [col for col in num_cols if col in df_scaled.columns]
        
        if fit:
            df_scaled[num_cols] = self.scaler.fit_transform(df_scaled[num_cols])
        else:
            df_scaled[num_cols] = self.scaler.transform(df_scaled[num_cols])
        
        return df_scaled
    
    def prepare_data(
        self, 
        filepath: str, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        """
        Complete data preparation pipeline: load, clean, engineer, encode, scale, and split.
        
        Args:
            filepath (str): Path to the CSV file
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple containing X_train, X_test, y_train, y_test, and exploration stats
        """
        logger.info("Starting complete data preparation pipeline")
        
        # Load data
        df = self.load_data(filepath)
        
        # Explore data
        stats = self.explore_data(df)
        
        # Clean data
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Separate features and target
        X = df.drop('enrolled', axis=1)
        y = df['enrolled']
        
        # Split data first (to prevent data leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Encode and scale training data (fit)
        X_train = self.encode_features(X_train, fit=True)
        X_train = self.scale_features(X_train, fit=True)
        
        # Encode and scale test data (transform only)
        X_test = self.encode_features(X_test, fit=False)
        X_test = self.scale_features(X_test, fit=False)
        
        # Store feature columns
        self.feature_columns = X_train.columns.tolist()
        
        logger.info("Data preparation complete")
        return X_train, X_test, y_train, y_test, stats
    
    def prepare_single_prediction(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare a single data point for prediction.
        
        Args:
            data (Dict): Dictionary containing feature values
            
        Returns:
            pd.DataFrame: Processed dataframe ready for prediction
        """
        df = pd.DataFrame([data])
        df = self.engineer_features(df)
        df = self.encode_features(df, fit=False)
        df = self.scale_features(df, fit=False)
        
        # Ensure columns match training data
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        return df[self.feature_columns]


if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor()
    X_train, X_test, y_train, y_test, stats = processor.prepare_data("employee_data.csv")
    
    print("\n=== Data Exploration Summary ===")
    print(f"Dataset Shape: {stats['shape']}")
    print(f"\nTarget Distribution:")
    print(f"  Not Enrolled (0): {stats['target_distribution'].get(0, 0)} ({stats['target_percentage'].get(0, 0):.1%})")
    print(f"  Enrolled (1): {stats['target_distribution'].get(1, 0)} ({stats['target_percentage'].get(1, 0):.1%})")
    print(f"\nEnrollment Rate by Employment Type:")
    for emp_type, rate in stats['enrollment_by_employment_type'].items():
        print(f"  {emp_type}: {rate:.1%}")
