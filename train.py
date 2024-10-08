import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

# Generate random data
# Let's assume we have 1000 samples and 10 features
X = np.random.rand(1000, 10)  # 1000 rows of 10 random feature columns
y = np.random.rand(1000)  # 1000 random target values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the experiment name
mlflow.set_experiment("Random_Data_Experiment")

# Start MLflow run
with mlflow.start_run():
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    
    # Log metrics
    mlflow.log_metric('mse_lr', mse_lr)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    
    # Log metrics
    mlflow.log_metric('mse_rf', mse_rf)
    import os

# Create the directory if it doesn't exist
    if not os.path.exists("/workspace"):
       os.makedirs("/workspace")
    # Log models
    mlflow.sklearn.log_model(lr, "Linear_Regression_Model")
    mlflow.sklearn.log_model(rf, "Random_Forest_Model")
    
    print(f"Linear Regression MSE: {mse_lr}")
    print(f"Random Forest MSE: {mse_rf}")

