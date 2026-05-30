import os
import multiprocessing
# FIX 1: Tell joblib exactly how many cores to use so it stops looking for 'wmic'
os.environ['LOKY_MAX_CPU_COUNT'] = str(multiprocessing.cpu_count() or 4)

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings

# Ignore lightgbm warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_preprocess():
    print("Loading datasets...")
    # Using your absolute paths
    train = pd.read_csv('C:/Users/sai/Downloads/e88186124ec611f1/dataset/train.csv')
    test = pd.read_csv('C:/Users/sai/Downloads/e88186124ec611f1/dataset/test.csv')
    
    # Keep track of lengths and test indices
    train_len = len(train)
    test_indices = test['Index'].copy()
    
    # Combine train and test to ensure consistent preprocessing
    combined = pd.concat([train, test], sort=False).reset_index(drop=True)
    
    print("Preprocessing data...")
    # 1. Time Features: Split 'timestamp' into hours and minutes
    combined['hour'] = combined['timestamp'].apply(lambda x: int(str(x).split(':')[0]) if pd.notnull(x) else 0)
    combined['minute'] = combined['timestamp'].apply(lambda x: int(str(x).split(':')[1]) if pd.notnull(x) else 0)
    combined.drop('timestamp', axis=1, inplace=True)
    
    # 2. Missing Values Handling
    combined['RoadType'] = combined['RoadType'].fillna('Unknown')
    combined['Weather'] = combined['Weather'].fillna('Unknown')
    combined['Temperature'] = combined['Temperature'].fillna(combined['Temperature'].median())
    combined['NumberofLanes'] = combined['NumberofLanes'].fillna(combined['NumberofLanes'].mode()[0])
    combined['LargeVehicles'] = combined['LargeVehicles'].fillna('Unknown')
    combined['Landmarks'] = combined['Landmarks'].fillna('Unknown')
    
    # 3. Categorical Encoding
    cat_cols = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for col in cat_cols:
        combined[col] = combined[col].astype(str)
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col])
        
    # Split back into train and test
    train_processed = combined.iloc[:train_len].copy()
    test_processed = combined.iloc[train_len:].copy()
    
    # Prepare X and y
    X = train_processed.drop(['Index', 'demand'], axis=1)
    y = train_processed['demand']
    X_test = test_processed.drop(['Index', 'demand'], axis=1)
    
    return X, y, X_test, test_indices

if __name__ == "__main__":
    # 1. Prepare Data
    X, y, X_test, test_indices = load_and_preprocess()
    
    # Split training data for Optuna evaluation
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Define Optuna Objective Function
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        
        # Train model
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        # FIX 2: Calculate RMSE using np.sqrt() (This prevents the 'squared' crash!)
        preds = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        return rmse

    # 3. Run Optuna Optimization
    print("\nStarting Optuna Hyperparameter Tuning...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30) 
    
    print('\nBest Hyperparameters found by Optuna:')
    print(study.best_params)

    # 4. Train Final Model on ALL Training Data
    print("\nTraining final model on full dataset with best parameters...")
    best_params = study.best_params
    best_params['objective'] = 'regression'
    best_params['random_state'] = 42
    best_params['verbosity'] = -1
    
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X, y)
    
    # 5. Generate Predictions and Save CSV
    print("Generating predictions for test data...")
    test_predictions = final_model.predict(X_test)
    
    # Create the submission dataframe
    submission = pd.DataFrame({
        'Index': test_indices,
        'demand': test_predictions
    })
    
    submission['demand'] = submission['demand'].clip(lower=0)
    print(X_test)
    # Saving safely to your downloads folder alongside the data
    output_path = 'C:/Users/sai/Downloads/e88186124ec611f1/dataset/submission.csv'
    submission.to_csv(output_path, index=False)
    print(f"\nSuccess! Predictions saved locally to: {output_path}")
