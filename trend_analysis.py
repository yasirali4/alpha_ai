import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load your data
data = pd.read_csv('properties.csv')

# Step 2: Preprocess the 'Location' column (convert to numeric)
encoder = LabelEncoder()
data['Location'] = encoder.fit_transform(data['Location'])

# Step 3: Convert 'Date' column to datetime and create 'Day of Year'
data['Date'] = pd.to_datetime(data['Date'])
data['Day_of_Year'] = data['Date'].dt.dayofyear

# Step 4: Prepare features (X) and target (y)
X = data[['Day_of_Year', 'Location', 'Size']]  # Now using 3 features
y = data['Price']  # Target is still the price

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 6: Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# Step 7: Support Vector Machine Model
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)

joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(svm_model,'models/svm_model.pkl')

# Step 8: Plot the results
plt.figure(figsize=(10, 6))

# Plot the actual prices
sns.lineplot(x='Date', y='Price', data=data, marker='o', label="Actual Prices")

# Plot the Random Forest predictions
rf_predicted_data = pd.DataFrame({'Date': data['Date'], 'Random Forest Predicted Price': rf_model.predict(X)})
sns.lineplot(x='Date', y='Random Forest Predicted Price', data=rf_predicted_data, marker='o', color='red', label="RF Predicted Prices")

# Plot the SVM predictions
svm_predicted_data = pd.DataFrame({'Date': data['Date'], 'SVM Predicted Price': svm_model.predict(X)})
sns.lineplot(x='Date', y='SVM Predicted Price', data=svm_predicted_data, marker='o', color='green', label="SVM Predicted Prices")

plt.title('Property Price Trend with Predictions (Random Forest and SVM)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 9: Evaluate Model Performance for Random Forest and SVM
print("Random Forest Model Evaluation:")
print("Mean Absolute Error:", mean_absolute_error(y_test, rf_y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, rf_y_pred))
print("R-squared:", r2_score(y_test, rf_y_pred))

print("\nSVM Model Evaluation:")
print("Mean Absolute Error:", mean_absolute_error(y_test, svm_y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, svm_y_pred))
print("R-squared:", r2_score(y_test, svm_y_pred))


