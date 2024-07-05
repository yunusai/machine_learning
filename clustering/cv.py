import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing functions
import re
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

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

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

# Fit K-means clustering model
# num_clusters = 2  # Define the number of clusters you want
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)

# Predict clusters
df['cluster'] = kmeans.labels_

# Evaluate the model using silhouette score
sil_score = silhouette_score(X, kmeans.labels_)
print("Silhouette Score:", sil_score)

# Print cluster labels
print(df)

# Visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
pca_df['cluster'] = df['cluster']

# Plotting the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=pca_df, palette='viridis', s=100, alpha=0.6, edgecolor='w', marker='o')
plt.title('Clusters visualization using PCA')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.show()
