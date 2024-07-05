import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Preprocessing functions
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

def preprocess(text):
    text = case_folding(text)
    tokens = tokenizing(text)
    text = stemming(tokens)
    text = stopword_removal(text)
    return text

# Load dataset from Excel file
df = pd.read_excel('20data.xlsx')

# Ensure the DataFrame has columns named 'text' and 'label'
# df = pd.read_csv('path_to_your_csv_file.csv', names=['text', 'label'])  # Use this if the CSV has no header row

# Apply preprocessing to the 'text' column
df['text'] = df['text'].apply(preprocess)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data, transform the test data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and fit the K-means model
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train_tfidf)

# Predict cluster labels on the test set
y_pred = kmeans.predict(X_test_tfidf)

# Evaluate clustering performance using silhouette score
silhouette_avg = silhouette_score(X_test_tfidf, y_pred)
print(f'Silhouette Score: {silhouette_avg}')

# Predict cluster labels on the training and test sets
y_pred_train = kmeans.predict(X_train_tfidf)
y_pred_test = kmeans.predict(X_test_tfidf)

# Assign cluster labels to the original DataFrame for visualization using PCA
df_train = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
df_train['cluster'] = y_pred_train  # Assigning predicted clusters to the training data

# Visualization using PCA for dimensionality reduction
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_tfidf.toarray())

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(X_train_pca, columns=['PCA1', 'PCA2'])
pca_df['cluster'] = y_pred_train  # Use predicted clusters for coloring

# Plotting the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=pca_df, palette='viridis', s=100, alpha=0.6, edgecolor='w', marker='o')
plt.title('Clusters visualization using PCA')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.show()