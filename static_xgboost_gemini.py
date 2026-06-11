import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np

# 1. Load the Data
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Assign the data folder dynamically from your config
PARQUET_PATH = config['static_dir'] + "/static_training_table.parquet"  # Update if your file name differs
df = pd.read_parquet(PARQUET_PATH)

# Drop rows with NaN responses or NaN climate predictors just in case
before = len(df); df = df.dropna(subset=['response'])
print(f"dropped {before-len(df)} rows with null response; kept {len(df)} (NaN predictors left for XGBoost)")

# 2. Separate Metadata, Predictors, and Target
metadata_cols = ['Grid_ID', 'year', 'cell_lat', 'cell_lon', 'n_events', 'mean_count', 'low_n_flag']
target_col = 'response'

# Assuming your climate features are all the remaining columns
predictor_cols = [col for col in df.columns if col not in metadata_cols and col != target_col]

X = df[predictor_cols]
y = df[target_col]
groups = df['year']         # For Leave-One-Year-Out CV
weights = df['n_events']    # For sample weighting

# 3. Setup XGBoost Regressor
# These are sensible baseline parameters for SDMs
xgb_model = xgb.XGBRegressor(
    n_estimators=300,        # Number of trees
    learning_rate=0.05,      # Step size shrinkage
    max_depth=5,             # Depth of tree (keep low to prevent spatial overfitting)
    subsample=0.8,           # Use 80% of rows per tree
    colsample_bytree=0.8,    # Use 80% of features per tree
    random_state=42,
    objective='reg:squarederror'
)

# 4. Cross-Validation (Leave-One-Year-Out)
logo = LeaveOneGroupOut()
oof_predictions = np.zeros(len(X)) # Out-of-fold predictions

print("Starting Leave-One-Year-Out Cross Validation...")

for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
    # Extract training and testing sets for this fold
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test   = X.iloc[test_idx], y.iloc[test_idx]
    
    # Extract weights for the training set
    w_train = weights.iloc[train_idx]
    
    # Get the year being left out for printing
    test_year = groups.iloc[test_idx].iloc[0]
    
    # Train the model with sample weights
    xgb_model.fit(X_train, y_train, sample_weight=w_train) 
    
    # Predict on the holdout year
    preds = xgb_model.predict(X_test)
    oof_predictions[test_idx] = preds
    
    # Calculate fold metric
    fold_r2 = r2_score(y_test, preds)
    print(f"Fold {fold+1} (Holdout Year: {test_year}): R-squared = {fold_r2:.3f}")

# 5. Overall Evaluation
overall_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
overall_r2 = r2_score(y, oof_predictions)

print("\n=== FINAL MODEL PERFORMANCE ===")
print(f"Overall RMSE: {overall_rmse:.3f} (in log1p units)")
print(f"Overall R-squared: {overall_r2:.3f}")

# 6. Feature Importance (Ecological View)
# Train one final model on ALL data to get final feature importances
xgb_model.fit(X, y, sample_weight=weights)

cells = df.groupby('Grid_ID')[['cell_lat','cell_lon']].first()
cells['block'] = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(cells)
df = df.merge(cells['block'], on='Grid_ID')
spatial_groups = df['block']  

importances = xgb_model.feature_importances_
feat_imp = pd.DataFrame({'Feature': predictor_cols, 'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=True).tail(10)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='skyblue')
plt.title("Top 10 Environmental Predictors for Cx. nigripalpus Abundance")
plt.xlabel("XGBoost Feature Importance (Gain)")
plt.tight_layout()
plt.show()