# Fraud Detection using Machine Learning

## Project Overview
This project aims to detect fraudulent transactions using machine learning techniques. The dataset used contains various features related to payment transactions, and the objective is to build a model that can accurately classify transactions as fraudulent or not.

## Table of Contents
Installation
Dataset
Exploratory Data Analysis
Data Preprocessing
Model Training
Model Evaluation
Results
Contributing
License
Installation
To run this project, you need to have Python installed along with the following libraries:

numpy
pandas
seaborn
matplotlib
scikit-learn

### About Dataset
The dataset used in this project is payment_fraud.csv. It contains the following columns:

accountAgeDays: Age of the account in days
numItems: Number of items purchased
localTime: Local time of the transaction
paymentMethod: Method of payment (e.g., PayPal, credit card)
paymentMethodAgeDays: Age of the payment method in days
label: Class label (0 for non-fraudulent, 1 for fraudulent)

### Exploratory Data Analysis
Initial data exploration includes visualizing the distribution of payment methods used in the transactions:

paymthd = df.paymentMethod.value_counts()
plt.figure(figsize=(5, 5))
sns.barplot(x = paymthd.index, y = paymthd)
plt.ylabel('Count')
plt.show()

## Data preprocessing steps include:

Handling missing values
Encoding categorical features
Scaling numerical features
## Model Training
The Logistic Regression model is used for training on the preprocessed data. The dataset is split into training and testing sets, and the model is trained as follows:

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#### Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#### Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#### Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

### Model Evaluation
The model is evaluated using metrics such as classification report, accuracy score, and confusion matrix:

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
### Results

The results of the model evaluation, including the accuracy, precision, recall, and F1-score, are presented to assess the model's performance in detecting fraudulent transactions.
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
