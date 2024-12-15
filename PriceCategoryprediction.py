import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

transactions = pd.read_json('transactions.json')

transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
transactions['DayOfWeek'] = transactions['TransactionDate'].dt.dayofweek
transactions['Month'] = transactions['TransactionDate'].dt.month
transactions['Hour'] = transactions['TransactionDate'].dt.hour

def price_category(amount):
    if amount < 100:
        return 'Low'
    elif 100 <= amount <= 300:
        return 'Medium'
    else:
        return 'High'

transactions['PriceRange'] = transactions['Amount'].apply(price_category)
transactions.to_csv('transactions_with_price_ranges.csv', index=False)

X = transactions[['DayOfWeek', 'Month', 'Hour', 'Category']]
y = transactions['PriceRange']

label_encoder_category = LabelEncoder()
X['Category'] = label_encoder_category.fit_transform(X['Category'])

label_encoder_price = LabelEncoder()
y = label_encoder_price.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder_price.classes_))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder_price.classes_, yticklabels=label_encoder_price.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
