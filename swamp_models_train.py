import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# Load data from Excel file
file_path = "./train.xlsx"
data = pd.read_excel(file_path)

# Process data
X = data.iloc[:, :-1]  # Feature columns
y = data.iloc[:, -1]   # Target column

# Normalize features using MinMaxScaler
scalerminmax = MinMaxScaler()
X_scaled = scalerminmax.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Create and train MLP model
mlp_model = MLPRegressor(
    hidden_layer_sizes=(200,100,25),
    activation='relu',  # Enhanced nonlinear activation
    solver='adam',      # Adaptive optimizer
    max_iter=2000,
    learning_rate='adaptive',  # Adaptive learning rate
    learning_rate_init=0.001,  # Learning rate
    alpha=0.0,          # Disable regularization (L2 penalty)
    random_state=42)

# Train MLP model on full normalized dataset
mlp_model.fit(X_scaled, y)
y_pred_mlp = mlp_model.predict(X_scaled)
print(f'Mean Squared Error: {mean_squared_error(y, y_pred_mlp)}')

# Create and train XGBoost model
xgb_model = xgb.XGBRegressor(
    n_estimators=2000,                    # Increase iterations
    learning_rate=0.01,                   # Reduce learning rate
    max_depth=12,                         # Increase tree depth
    min_child_weight=5,                   # Control minimum sample weight in leaf nodes
    subsample=0.7,                        # Row sampling ratio
    colsample_bytree=0.7,                 # Column sampling ratio
    reg_lambda=1.5,                       # L2 regularization
    reg_alpha=0.5,                        # L1 regularization
    objective='reg:squarederror',         # Regression task
    tree_method='hist',                   # Histogram optimization acceleration
    enable_categorical=True,              # Enable categorical feature support
    random_state=42,
    scale_pos_weight=1.0                  # Handle class imbalance (if exists)
)

# Train XGBoost model on training set
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print(f'XGBoost MSE: {mean_squared_error(y_test, y_pred_xgb):.4f}')

# Save MLP model
joblib.dump(mlp_model, 'swamp_mlp_model.pkl')

# Save XGBoost model
joblib.dump(xgb_model, 'swamp_xgboost_model.pkl')