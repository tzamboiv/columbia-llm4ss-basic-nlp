import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import nltk
from nltk.stem import PorterStemmer
#may need to run following if you get a spacy error
#python -m spacy download en_core_web_sm
df = pd.read_csv("CommentsMay2017.csv")
df = df.rename(columns={'commentBody': 'text'})

# 1. Global Initialization
# Initialize these outside the function so they don't reload on every function call
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
stemmer = PorterStemmer()

def modelTopics(df, n_components, preprocessing_method="lemmatize"):
    """
    Fits an LDA model to a dataframe of texts.

    Parameters:
    - df: pandas DataFrame containing a 'text' column.
    - n_components: int, the number of topics (K) to extract.
    - preprocessing_method: str, either "lemmatize" (slower, higher quality) or "stem" (faster, lower quality).
    """
    vectorizer = CountVectorizer(stop_words="english", max_df=0.95, min_df=10)

    # 2. The Preprocessing Flag
    if preprocessing_method == "lemmatize":
        print("Lemmatizing text in parallel (this will take longer)...")
        articles = [
            " ".join([token.lemma_ for token in doc])
            for doc in nlp.pipe(df['text'].astype(str), n_process=-1, batch_size=100)
        ]

    elif preprocessing_method == "stem":
        print("Stemming text (this will be extremely fast)...")
        # Use the vectorizer's analyzer to handle lowercasing and punctuation stripping cleanly
        analyzer = vectorizer.build_analyzer()
        articles = [
            " ".join([stemmer.stem(word) for word in analyzer(text)])
            for text in df['text'].astype(str)
        ]

    else:
        raise ValueError('preprocessing_method must be either "lemmatize" or "stem"')

    # 3. Model Training
    print("Preprocessing complete. Building document-term matrix...")
    doc_term_matrix = vectorizer.fit_transform(articles)

    print(f"Matrix created. Fitting LDA model for {n_components} topics...")
    lda = LatentDirichletAllocation(
        n_components=n_components,
        random_state=42,
        n_jobs=-1
    )

    topic_distributions = lda.fit_transform(doc_term_matrix)

    # 4. Results Merging
    print("LDA model fitted. Merging results...")
    topic_columns = [f"Topic_{i}" for i in range(lda.n_components)]
    topic_df = pd.DataFrame(topic_distributions, columns=topic_columns, index=df.index)

    final_df = pd.concat([df, topic_df], axis=1)
    final_df['dominant_topic'] = np.argmax(topic_distributions, axis=1)

    return lda, final_df

print(modelTopics(df, 10, "stem"))
