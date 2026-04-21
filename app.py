import numpy as np
import pandas as pd
import re
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

app = Flask(__name__)

# DATASET
documents = [
    "Artificial intelligence is transforming technology",
    "Machine learning is a subset of artificial intelligence",
    "Deep learning improves AI performance",
    "Data science involves statistics and machine learning",
    "Big data analytics is important in modern technology",
    "AI is used in healthcare and finance",
    "Neural networks are used in deep learning",
    "Technology is evolving rapidly with AI",
    "Machine learning models require data",
    "Artificial intelligence and data science are related"
]

stopwords = {"is", "a", "and", "in", "of", "the", "are", "with"}

def preprocess(text):
    text = text.lower()                         # lowercasing
    text = re.sub(r'[^a-z\s]', '', text)        # hapus simbol
    words = text.split()
    words = [w for w in words if w not in stopwords]  # hapus stopwords
    return " ".join(words)

documents_clean = [preprocess(doc) for doc in documents]

# TF-IDF Setup
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents_clean)
feature_names = vectorizer.get_feature_names_out()

# Calculate manual metrics for display
def compute_tf(doc):
    words = doc.split()
    tf = Counter(words)
    total_words = len(words)
    if total_words == 0: return {}
    return {word: count / total_words for word, count in tf.items()}

def compute_idf(docs):
    N = len(docs)
    idf = {}
    all_words = set(word for doc in docs for word in doc.split())
    for word in all_words:
        df = sum(1 for doc in docs if word in doc)
        idf[word] = math.log(N / (df + 1))  # tambah +1 biar aman
    return idf

def compute_tfidf(tf, idf):
    return {word: tf[word] * idf.get(word, 0) for word in tf}

tf_docs = [compute_tf(doc) for doc in documents_clean]
idf = compute_idf(documents_clean)
tfidf_manual = [compute_tfidf(tf, idf) for tf in tf_docs]

tf_df = pd.DataFrame(tf_docs).fillna(0)
tfidf_df = pd.DataFrame(tfidf_manual).fillna(0)
tfidf_df_lib = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Convert to HTML to render in templates
def format_df_to_html(df):
    return df.round(3).to_html(
        classes='table-display',
        border=0,
        justify='left'
    )

tf_html = format_df_to_html(tf_df)
idf_html = pd.DataFrame(list(idf.items()), columns=['Word', 'IDF']).round(3).to_html(
    classes='table-display', border=0, justify='left', index=False
)
tfidf_html = format_df_to_html(tfidf_df)
tfidf_lib_html = format_df_to_html(tfidf_df_lib)

def search_query(query, top_k=5):
    query_clean = preprocess(query)
    if not query_clean.strip():
        return None

    query_vec = vectorizer.transform([query_clean])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    scores = similarity.flatten()
    
    ranked_indices = scores.argsort()[::-1]
    
    results = []
    for rank, i in enumerate(ranked_indices[:top_k], start=1):
        if scores[i] > 0: # Only return somewhat matching
            results.append({
                "Rank": f"#{rank}",
                "Document": documents[i],
                "Score": f"{scores[i]:.4f}"
            })
    return pd.DataFrame(results)

@app.route('/', methods=['GET', 'POST'])
def index():
    query = ""
    search_results = None
    
    if request.method == 'POST':
        query = request.form.get('query', '')
        if query:
            result_df = search_query(query)
            if result_df is not None and not result_df.empty:
                search_results = result_df.to_html(classes='table-display search-results-table', border=0, index=False, justify='left')
            else:
                search_results = "<div class='no-results'>No relevant documents found for your query.</div>"

    return render_template(
        'index.html',
        documents=enumerate(documents, 1),
        tf_html=tf_html,
        idf_html=idf_html,
        tfidf_html=tfidf_html,
        tfidf_lib_html=tfidf_lib_html,
        query=query,
        search_results=search_results
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
