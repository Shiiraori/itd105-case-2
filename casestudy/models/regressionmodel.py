import pandas as pd
from scipy.stats.mstats import winsorize

# if need to reaccess data saved from google drive
# from google.colab import drive
# drive.mount('/content/drive')
# df = pd.read_csv('/content/drive/My Drive/School/4th year 1st sem/classification_sepsis_survival_prediction.csv')
df = pd.read_csv('regression_Real_estate.csv')

removecol = ['transaction date', 'latitude', 'longitude']
df = df.drop(removecol, axis=1)



# winzorcol = ['serum_creatinine', 'platelets', 'creatinine_phosphokinase']
df['distance to the nearest MRT station'] = winsorize(df['distance to the nearest MRT station'], limits=[0.09, 0.09])
df['house price of unit area'] = winsorize(df['house price of unit area'], limits=[0.05, 0.05])

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import joblib  # Use joblib for model persistence

# Separate features (X) and target variable (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Number of folds for KFold cross-validation
num_folds = 5

# Initialize KFold
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto')  # You can adjust the number of neighbors as needed

mae_scores = []

# Perform KFold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    knn_model.fit(X_train, y_train)

    # Make predictions
    y_pred = knn_model.predict(X_test)

    # Evaluate with MAE
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae)

# Calculate the average MAE across all folds
average_mae = sum(mae_scores) / num_folds

print(f'Average MAE: {average_mae}')

# Save the trained model to a file using joblib
model_filename = 'regressionmodel.pkl'
joblib.dump(knn_model, model_filename)
print(f'Model saved as {model_filename}')