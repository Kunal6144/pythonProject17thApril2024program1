import pandas as pd

# Read the data into pandas dataframe
df = pd.read_csv('Auto.csv')

# Define X and y
X = df.drop(columns=['mpg', 'name', 'origin'])
y = df['mpg']

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import Ridge, Lasso

# Initialize lists to store R2 scores
ridge_scores = []
lasso_scores = []

# Define a range of alpha values to try
alphas = [0.1, 0.5, 1, 5, 10, 50, 100]

# Loop through alpha values and fit models
for alpha in alphas:
    # Initialize ridge regression model
    ridge_model = Ridge(alpha=alpha)
    # Fit ridge regression model
    ridge_model.fit(X_train, y_train)
    # Compute R2 score and append to list
    ridge_scores.append(ridge_model.score(X_test, y_test))

    # Initialize LASSO regression model
    lasso_model = Lasso(alpha=alpha)
    # Fit LASSO regression model
    lasso_model.fit(X_train, y_train)
    # Compute R2 score and append to list
    lasso_scores.append(lasso_model.score(X_test, y_test))

import matplotlib.pyplot as plt

# Plot R2 scores for ridge regression
plt.plot(alphas, ridge_scores, label='Ridge Regression')

# Plot R2 scores for LASSO regression
plt.plot(alphas, lasso_scores, label='LASSO Regression')

# Add labels and title
plt.xlabel('Alpha')
plt.ylabel('R2 Score')
plt.title('R2 Score vs. Alpha')
plt.legend()
plt.show()
