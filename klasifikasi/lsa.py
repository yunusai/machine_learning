# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Preprocessing functions
import re
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

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

# Define an LSA pipeline with CountVectorizer, TruncatedSVD, and Normalizer
lsa_pipeline = make_pipeline(CountVectorizer(), 
                             TruncatedSVD(n_components=85, random_state=42), 
                             Normalizer(copy=False))

# Fit and transform on training data
X_train_lsa = lsa_pipeline.fit_transform(X_train)
X_test_lsa = lsa_pipeline.transform(X_test)

import pandas as pd

# Get the components from LSA pipeline
vectorizer = lsa_pipeline.named_steps['countvectorizer']
svd = lsa_pipeline.named_steps['truncatedsvd']
lsa_components = svd.components_

# Prepare data for tabular display
data = []
feature_names = vectorizer.get_feature_names_out()

# Create a DataFrame for transformed LSA data
df_lsa_transformed = pd.DataFrame(X_train_lsa, columns=[f'Component {i+1}' for i in range(lsa_components.shape[0])])

# Print LSA Transformed Data
print("LSA Transformed Data:")
print(df_lsa_transformed.head())

# Print explained variance ratio
print("\nExplained Variance Ratio:", svd.explained_variance_ratio_)

# Print the shape of LSA components
print("\nLSA Components Shape:", lsa_components.shape)

# Prepare data for components table
component_data = []

# Iterate over each component
for i, component in enumerate(lsa_components):
    component_terms = []
    for j, term in enumerate(feature_names):
        component_terms.append((term, component[j]))
    component_data.append(pd.DataFrame(component_terms, columns=['Term', f'Component {i+1}']))

# Concatenate all components into a single DataFrame
df_components = pd.concat(component_data, axis=1)

# Display the components table
print("\nComponents Table:")
print(df_components)


# Train Logistic Regression model on LSA-transformed data
model = LogisticRegression()
model.fit(X_train_lsa, y_train)

# Predict on test data
y_pred = model.predict(X_test_lsa)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
