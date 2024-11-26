from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

file_path = 'DS-GA-1007/Zeyuan/student_data.csv'
data = pd.read_csv(file_path)

# Display the first few rows to understand the structure of the data
data.head()

data = data.drop(columns=['G1', 'G2'])

# Remove G1 and G2 from the features and define X and y
X = data.drop(columns=['G3'])  # Remove G1, G2, and G3 (target variable)
y = data['G3']  # Target variable

# Convert categorical data to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse, r2

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

# Calculate the correlation matrix to find features most correlated with G3
data = pd.get_dummies(data, drop_first=True)

correlation_matrix = data.corr()
correlated_features = correlation_matrix['G3'].abs().sort_values(ascending=False).index

# Select top features based on correlation (excluding G3 itself)
top_features = correlated_features[1:11]  # Select top 10 correlated features (excluding G3)
X = data[top_features]
print(top_features)

# Convert categorical variables to numeric using one-hot encoding if necessary
X = pd.get_dummies(X, drop_first=True)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(mse_rf, r2_rf)
