import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic


#Data
df = pd.read_csv("CommentsMay2017.csv")
docs = df.commentBody.tolist()[:10000]


# Embedding
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(docs, show_progress_bar=True)


# Dimensionality Reduction


umap_model = UMAP(
    n_neighbors=15,       # Increase to look at broader global structure
    n_components=5,       # Dimensions to reduce to
    min_dist=0.0,         # Controls how tightly UMAP packs points together
    metric='cosine',
    random_state=42       # Set for reproducibility
)


# Clustering

hdbscan_model = HDBSCAN(
    min_cluster_size=5,  # Minimum documents needed to form a topic
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

#Vectorizer
# You can add custom stop word lists here relevant to specific institutional jargon
vectorizer_model = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),   # Allow unigrams and bigrams
    min_df=2              # Ignore words that appear in fewer than 2 documents
)

ctfidf_model = ClassTfidfTransformer(
    reduce_frequent_words=True, # Penalizes words that appear in many topics
    bm25_weighting=False
)


# Topic Model
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model
)

topics, probabilities = topic_model.fit_transform(docs, embeddings=embeddings)
topic_model.get_topic_info().to_csv("topicMOdel.csv")
print(topic_model.get_topic_info())
