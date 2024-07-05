# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load data into a Pandas DataFrame
df = pd.read_excel('20data.xlsx')

# Assuming your Excel has columns 'text' for text data and 'label' for labels
X = df['text'].astype(str)  # Convert text column to string type
y = df['label']  # Replace 'label' with your actual label column name

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the data
X = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize logistic regression model
logreg = LogisticRegression(max_iter=1000)

# Fit the model on the training data
logreg.fit(X_train, y_train)

# Predict on the test data
y_pred = logreg.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy}")

# Print classification report
print(metrics.classification_report(y_test, y_pred))
