import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the prepared dataset
data = pd.read_csv('churn_data_for_prediction.csv')

# Step 2: Handle missing values (fill NA values for categorical columns)
data['PreferredChannel'].fillna('Unknown', inplace=True)
data['AvgResolutionTime'].fillna(data['AvgResolutionTime'].mean(), inplace=True)
data['InteractionCount'].fillna(0, inplace=True)

# Step 3: Encode categorical features
data = pd.get_dummies(data, columns=['PreferredChannel'], drop_first=True)

# Step 4: Define features and target variable
X = data.drop(columns=['CustomerID', 'FirstName', 'LastName', 'Email', 'Phone', 'Churn'])
y = data['Churn']
missing_features = X.isnull().sum()
print(missing_features[missing_features > 0])
# Step 5: Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\nClassification Report:\n", class_report)

# Step 8: Visualization - Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Active', 'Churned'], yticklabels=['Active', 'Churned'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 9: Feature Importance
# Logistic Regression coefficients indicate feature importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.coef_[0]
}).sort_values(by='Importance', ascending=False)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance')
plt.show()
