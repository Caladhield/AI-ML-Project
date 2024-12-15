import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def save_to_csv(data, filename):
    data.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

transactions = pd.read_json('transactions.json')
customer_details = pd.read_csv('customer_details.csv')

#Join data to see if transaction data matches an actual customer
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

#All data per customor
aggregated_data = transactions.groupby('CustomerID').agg({
    'Amount': 'sum',  # TotalSpent
    'TransactionID': 'count',  # Frequency
    'TransactionDate': lambda x: (datetime.now() - x.max()).days
}).rename(columns={
    'Amount': 'TotalSpent',
    'TransactionID': 'Frequency',
    'TransactionDate': 'Recency'
}).reset_index()

data = pd.merge(customer_details, aggregated_data, on='CustomerID', how='left')


data['TotalSpent'] = data['TotalSpent'].fillna(0)
data['Frequency'] = data['Frequency'].fillna(0)
data['Recency'] = data['Recency'].fillna(data['Recency'].max())


data['HighValue'] = np.where(data['TotalSpent'] > 350, 1, 0) #If totalspent is higher tahn 350, label with 1, otherwise 0

save_to_csv(data, 'processed_high_value_customers.csv')

X = data[['TotalSpent', 'Frequency', 'Recency']]
y = data['HighValue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low-Value', 'High-Value'], yticklabels=['Low-Value', 'High-Value'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 3. Bar Chart: High vs. Low-Value Customers
plt.figure(figsize=(8, 6))
sns.countplot(x='HighValue', data=data, palette='Set2')
plt.title('High-Value vs. Low-Value Customers')
plt.xlabel('Customer Type (0 = Low, 1 = High)')
plt.ylabel('Count')
plt.show()

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()
