# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score, classification_report

# Preprocessing functions
import re
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Preprocessing function
# Case folding function
def case_folding(text):
    return text.lower()

# Tokenizing function
def tokenizing(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text)

# Stemming function
def stemming(tokens):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(' '.join(tokens))

# Stopword removal function
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

# Tagging documents for Doc2Vec
tagged_data_train = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(X_train)]
tagged_data_test = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(X_test)]

# Initialize Doc2Vec model
model_d2v = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=20)

# Build vocabulary
model_d2v.build_vocab(tagged_data_train)

# Train Doc2Vec model
model_d2v.train(tagged_data_train, total_examples=model_d2v.corpus_count, epochs=model_d2v.epochs)

# Vectorize the documents
X_train_vec = [model_d2v.infer_vector(doc.words) for doc in tagged_data_train]
X_test_vec = [model_d2v.infer_vector(doc.words) for doc in tagged_data_test]

# Convert vectors to DataFrame for inspection
df_train_vec = pd.DataFrame(X_train_vec)
df_test_vec = pd.DataFrame(X_test_vec)

# Display a few rows of the extracted vectors
print("Doc2Vec extracted vectors (Training Data):")
print(df_train_vec.head())

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict on test data
y_pred = model.predict(X_test_vec)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
