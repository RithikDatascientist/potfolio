import yaml
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'models', 'model_config.yaml')

with open(CONFIG_PATH, 'r') as file:
    xg = yaml.safe_load(file)



def select_features(x, y, method='xgboost', top_n=20):
    """Select the top N features using feature importance"""
    
    if method == 'xgboost':
        model = XGBRegressor(**xg['xgboost'])
    elif method == 'randomforest':
        model = RandomForestRegressor(**xg['randomforest'])
    
    model.fit(x, y)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': x.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance.head(top_n)['feature'].tolist()


def get_best_features(x, y, top_n=20):
    """Get features that are important in both models"""
    
    # Get top features from both models
    xgb_features = select_features(x, y, method='xgboost', top_n=top_n)
    rf_features = select_features(x, y, method='randomforest', top_n=top_n)
    
    # Features that appear in both (intersection)
    common_features = list(set(xgb_features) & set(rf_features))
    
    # All important features from both (union)
    all_features = list(set(xgb_features + rf_features))
    
    print(f"XGBoost selected {len(xgb_features)} features")
    print(f"RandomForest selected {len(rf_features)} features")
    print(f"Common features: {len(common_features)}")
    print(f"Total unique features: {len(all_features)}")
    
    return {
        'xgboost': xgb_features,
        'randomforest': rf_features,
        'common': common_features,
        'all': all_features
    }