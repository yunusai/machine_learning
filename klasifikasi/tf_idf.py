import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset from CSV file
df = pd.read_excel('20data.xlsx')

# Ensure the CSV has columns named 'text' and 'label'
# df = pd.read_csv('path_to_your_csv_file.csv', names=['text', 'label'])  # Use this if the CSV has no header row

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data, transform the test data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and train the logistic regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train_tfidf, y_train)

# Predict the labels on the test set
y_pred = logistic_regression.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
