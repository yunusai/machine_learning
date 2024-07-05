import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

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

# Define LSA (TruncatedSVD) pipeline
lsa_pipeline = make_pipeline(CountVectorizer(),
                             TruncatedSVD(n_components=100, random_state=42),
                             Normalizer(copy=False))

# Fit LSA pipeline on the data and transform
X_lsa = lsa_pipeline.fit_transform(df['text'])

# Apply K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_lsa)

# Compute silhouette score
silhouette_avg = silhouette_score(X_lsa, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg}")

# Visualize clustering results using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_lsa)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
pca_df['cluster'] = kmeans.labels_

# Plotting the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=pca_df, palette='viridis', s=100, alpha=0.6, edgecolor='w', marker='o')
plt.title('Clusters Visualization using PCA')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.show()
