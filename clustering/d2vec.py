import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns  # Import seaborn for plotting

# Preprocessing functions
import re
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

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

# Train K-means model
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train_vec)

# Predict clusters for the test data
y_pred_train = kmeans.predict(X_train_vec)
y_pred_test = kmeans.predict(X_test_vec)

# Visualization using PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_vec)
X_test_pca = pca.transform(X_test_vec)

# Create a DataFrame for PCA results
pca_df_train = pd.DataFrame(X_train_pca, columns=['PCA1', 'PCA2'])
pca_df_train['cluster'] = y_pred_train

pca_df_test = pd.DataFrame(X_test_pca, columns=['PCA1', 'PCA2'])
pca_df_test['cluster'] = y_pred_test

# Plotting the training set clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=pca_df_train, palette='viridis', s=100, alpha=0.6, edgecolor='w', marker='o')
plt.title('Clusters Visualization using PCA (Training Set)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.show()

# Plotting the test set clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=pca_df_test, palette='viridis', s=100, alpha=0.6, edgecolor='w', marker='o')
plt.title('Clusters Visualization using PCA (Test Set)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.show()

# Evaluate the model with silhouette score on the test set
silhouette_avg = silhouette_score(X_test_vec, y_pred_test)
print("Silhouette Score:", silhouette_avg)
