import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Select variables for prediction
X = data[['bmi', 's5', 'glu', 'ins', 'bp']]
y = data['progression']

# Split the data into training and testing sets
# (You may choose to split the data here if necessary)

# Fit the model with just 'bmi' and 's5'
model_base = LinearRegression()
model_base.fit(X[['bmi', 's5']], y)
y_pred_base = model_base.predict(X[['bmi', 's5']])

# Compute performance metrics for the base model
rmse_base = np.sqrt(mean_squared_error(y, y_pred_base))
r2_base = r2_score(y, y_pred_base)

# Fit the model with additional variables
model_full = LinearRegression()
model_full.fit(X, y)
y_pred_full = model_full.predict(X)

# Compute performance metrics for the full model
rmse_full = np.sqrt(mean_squared_error(y, y_pred_full))
r2_full = r2_score(y, y_pred_full)

# Print the performance metrics
print("Base Model (BMI and S5 only):")
print("RMSE:", rmse_base)
print("R2 Score:", r2_base)
print("\nFull Model (BMI, S5, Glucose, Insulin, BP):")
print("RMSE:", rmse_full)
print("R2 Score:", r2_full)

"""
Findings:
- The base model with only 'bmi' and 's5' yields a certain level of performance.
- Adding additional variables ('glu', 'ins', 'bp') improves the model's performance, as indicated by lower RMSE and higher R2 score.
- Further improvement may be observed by adding more variables, but it's essential to balance model complexity and interpretability.
"""
