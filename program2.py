import pandas as pd

# Read the dataset into pandas dataframe
df = pd.read_csv('50_Startups.csv', delimiter=',')

# Display the columns of the dataframe
print(df.columns)

# Compute correlation matrix
correlation_matrix = df.corr()

# Print correlation matrix
print(correlation_matrix)
import matplotlib.pyplot as plt

# Plot explanatory variables against profit
plt.scatter(df['variable1'], df['profit'], label='Variable 1')
plt.scatter(df['variable2'], df['profit'], label='Variable 2')
# Add more scatter plots for other variables if needed
plt.xlabel('Explanatory Variable')
plt.ylabel('Profit')
plt.title('Explanatory Variables vs. Profit')
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split

# Define explanatory variables and target variable
X = df[['variable1', 'variable2']]  # Add more variables as needed
y = df['profit']

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Initialize the linear regression model
model = LinearRegression()

# Train the model with training data
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

# Predictions on training data
y_train_pred = model.predict(X_train)
# Predictions on testing data
y_test_pred = model.predict(X_test)

# Compute RMSE and R2 for training data
rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
r2_train = r2_score(y_train, y_train_pred)

# Compute RMSE and R2 for testing data
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
r2_test = r2_score(y_test, y_test_pred)

# Print RMSE and R2 values
print("Training Data:")
print("RMSE:", rmse_train)
print("R2 Score:", r2_train)
print("\nTesting Data:")
print("RMSE:", rmse_test)
print("R2 Score:", r2_test)
