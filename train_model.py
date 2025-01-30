import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(filepath: str):
    """Load and preprocess the crane dataset."""
    try:
        df = pd.read_csv(filepath)
        logger.info("Data loaded successfully")
        
        # Select features
        features = ['max_load', 'radius', 'wind_tolerance', 'load_weight', 'wind_speed']
        X = df[features]
        y = df['safe']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaler for later use
        joblib.dump(scaler, 'scaler.joblib')
        
        return X_scaled, y
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def train_model(X, y):
    """Train and optimize the RandomForest model."""
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        
        # Initialize model
        rf = RandomForestClassifier(random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1')
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate model
        y_pred = best_model.predict(X_test)
        logger.info("\nModel Performance Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Save model
        joblib.dump(best_model, 'crane_model.joblib')
        logger.info("Model saved successfully")
        
        return best_model
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data('crane_data.csv')
        
        # Train model
        model = train_model(X, y)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}") 