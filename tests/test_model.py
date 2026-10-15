import pytest
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

# Assume your model pipeline creation logic is encapsulated in a module (e.g., src.model_pipeline)
# For this test, we recreate a minimal version of the preprocessing logic
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def sample_data():
    """Fixture to provide a small sample of mock patient data."""
    data = {
        'age': [55, 60, 45, 50],
        'sex': [1, 0, 1, 0],
        'cp': [3, 2, 1, 0],
        'trestbps': [140, 130, 120, 110],
        'chol': [250, 200, 220, 240],
        'fbs': [0, 0, 1, 0],
        'restecg': [1, 0, 1, 0],
        'thalach': [150, 160, 140, 130],
        'exang': [0, 1, 0, 1],
        'oldpeak': [1.5, 2.0, 0.5, 1.0],
        'slope': [1, 2, 1, 0],
        'ca': [0, 1, 0, 2],
        'thal': [2, 3, 2, 1]
    }
    targets = [1, 1, 0, 0]
    return pd.DataFrame(data), np.array(targets)

@pytest.fixture
def mock_pipeline():
    """Fixture to provide an untrained model pipeline."""
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
    ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=10))
    ])
    return pipeline

def test_preprocessing_pipeline(sample_data, mock_pipeline):
    """Test that data preprocessing transforms inputs without dropping rows."""
    X, y = sample_data
    preprocessor = mock_pipeline.named_steps['preprocessor']
    
    X_transformed = preprocessor.fit_transform(X)
    
    # Assert output is generated and matches expected row count
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] > X.shape[1]  # One-hot encoding expands columns

def test_model_training_and_inference(sample_data, mock_pipeline):
    """Test that the full pipeline can train and predict valid binary outputs."""
    X, y = sample_data
    
    # Ensure it raises an error if predict is called before fit
    with pytest.raises(NotFittedError):
        mock_pipeline.predict(X)
        
    mock_pipeline.fit(X, y)
    predictions = mock_pipeline.predict(X)
    
    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})  # Binary classification constraint