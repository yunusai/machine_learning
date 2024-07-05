import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# Define preprocessing functions
def case_folding(text):
    return text.lower()

def tokenizing(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text)

def stemming(tokens):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(' '.join(tokens))

def stopword_removal(text):
    wordlist = set(stopwords.words('indonesian'))
    return ' '.join([word for word in text.split() if word not in wordlist])

# Full preprocessing pipeline
def preprocess(text):
    text = case_folding(text)
    tokens = tokenizing(text)
    text = stemming(tokens)
    text = stopword_removal(text)
    return text

# Load dataset from Excel file
df = pd.read_excel('20data.xlsx')

# Apply preprocessing to the 'text' column
df['text'] = df['text'].apply(preprocess)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data, transform the test data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Get feature names (words) and IDF values from TF-IDF vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()
idf_values = tfidf_vectorizer.idf_

# Create a DataFrame for feature names and IDF values
features_df = pd.DataFrame({'Feature Name': feature_names, 'IDF': idf_values})

# Print feature extraction details in a tabular format
print(f"=== Feature Extraction Details ===")
print(f"Shape of TF-IDF matrix: {X_train_tfidf.shape}")
print(features_df.to_string(index=False))

# Initialize and train the logistic regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train_tfidf, y_train)

# Predict the labels on the test set
y_pred = logistic_regression.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nModel Evaluation:")
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
