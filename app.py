import numpy as np
import pandas as pd
import re
import math
import os
import urllib.parse
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, send_from_directory
from PyPDF2 import PdfReader

app = Flask(__name__)

# DATASET
dataset_dir = 'dataset_pdf'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

documents = []
document_names = []

for filename in os.listdir(dataset_dir):
    if filename.endswith('.pdf'):
        filepath = os.path.join(dataset_dir, filename)
        try:
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
            if text.strip():
                documents.append(text.strip())
                document_names.append(filename)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Fallback dataset jika tidak ada file PDF
if not documents:
    print("Tidak ditemukan file PDF di folder 'dataset_pdf'. Menggunakan dataset default.")
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
    document_names = [f"Doc_{i+1}" for i in range(len(documents))]

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

def get_snippet(text, query, context_range=60):
    query_clean = preprocess(query)
    terms = [term for term in set(query_clean.split()) if term]
    
    if not terms:
        return text[:150] + "..."
    
    text_lower = text.lower()
    first_idx = -1
    for term in terms:
        idx = text_lower.find(term)
        if idx != -1:
            if first_idx == -1 or idx < first_idx:
                first_idx = idx
    
    if first_idx == -1:
        return text[:150] + "..."
        
    start = max(0, first_idx - context_range)
    end = min(len(text), first_idx + context_range + len(query))
    
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
        
    # Bold the terms
    for term in terms:
        pattern = re.compile(f"({re.escape(term)})", re.IGNORECASE)
        snippet = pattern.sub(r"<b style='color: #58a6ff; font-weight: 800;'>\1</b>", snippet)
        
    return snippet

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
            preview_html = get_snippet(documents[i], query_clean)
            
            search_param = urllib.parse.quote(query.strip())
            doc_name_encoded = urllib.parse.quote(document_names[i])
            
            results.append({
                "Rank": f"#{rank}",
                "Document": f"<strong style='color:#fff;'>[{document_names[i]}]</strong><br/>{preview_html}",
                "Score": f"{scores[i]:.4f}",
                "Action": f"<a href='/pdf/{doc_name_encoded}?q={search_param}' target='_blank' style='display:inline-block; margin-top:4px; padding:4px 10px; background:var(--surface-light); border:1px solid var(--primary); border-radius:6px; color:var(--primary); text-decoration:none; font-size:0.85rem; transition:0.2s;' onmouseover='this.style.background=\"var(--primary)\"; this.style.color=\"#fff\";' onmouseout='this.style.background=\"var(--surface-light)\"; this.style.color=\"var(--primary)\";'>📄 View PDF</a>"
            })
    return pd.DataFrame(results)

@app.route('/pdf/<filename>')
def serve_pdf(filename):
    query = request.args.get('q', '')
    filepath = os.path.join(dataset_dir, filename)
    
    if query:
        try:
            import fitz  # PyMuPDF
            import io
            from flask import send_file
            
            doc = fitz.open(filepath)
            query_terms = [t for t in query.split() if t.strip()]
            
            # Cari dan highlight tiap kata di semua halaman PDF
            for page in doc:
                for term in query_terms:
                    # search_for secara otomatis case-insensitive
                    text_instances = page.search_for(term)
                    for inst in text_instances:
                        annot = page.add_highlight_annot(inst)
                        annot.update()
                        
            pdf_bytes = doc.write()
            return send_file(io.BytesIO(pdf_bytes), mimetype='application/pdf')
            
        except ImportError:
            print("PyMuPDF tidak terinstall, mengirimkan PDF biasa sebagai fallback.")
        except Exception as e:
            print(f"Gagal melakukan highlight pada PDF: {e}")

    # Fallback ke PDF biasa jika tidak ada query atau PyMuPDF error
    return send_from_directory(dataset_dir, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    query = ""
    search_results = None
    
    if request.method == 'POST':
        query = request.form.get('query', '')
        if query:
            result_df = search_query(query)
            if result_df is not None and not result_df.empty:
                search_results = result_df.to_html(classes='table-display search-results-table', border=0, index=False, justify='left', escape=False)
            else:
                search_results = "<div class='no-results'>No relevant documents found for your query.</div>"

    # Format document preview for display in the UI
    documents_display = []
    for name, doc in zip(document_names, documents):
        preview = doc[:150] + "..." if len(doc) > 150 else doc
        documents_display.append({"name": name, "preview": f"[{name}] {preview}"})

    return render_template(
        'index.html',
        documents=enumerate(documents_display, 1),
        tf_html=tf_html,
        idf_html=idf_html,
        tfidf_html=tfidf_html,
        tfidf_lib_html=tfidf_lib_html,
        query=query,
        search_results=search_results
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
