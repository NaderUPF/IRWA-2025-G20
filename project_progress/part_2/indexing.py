"""
1. Build inverted index: After having pre-processed the data, you can then create the inverted index. 

HINT - you may use the vocabulary data structure, like the one seen during the Practical Labs: 
{ 
    Term_id_1: [document_1, document_2, document_4], 
    Term_id_2: [document_1, document_3, document_5, document_6], etc… 
}    
    
Important: For this assignment, we will be using conjunctive queries (AND). This means that 
every returned document must contain all the words from the query in order to be considered a match. 

"""

import json                # para leer y guardar archivos JSON
import math                # para logaritmos y cálculos de IDF
import os                  # para manejo de rutas y creación de carpetas
from collections import defaultdict, Counter  # para contar y construir índices
from typing import Dict, List, Tuple           # para anotaciones de tipos
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from project_progress.part_1.preprocess import _clean_text, _tokenize_and_normalize  # para limpiar y normalizar
from project_progress.part_1.preprocess import _clean_text  # para limpiar el texto
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def load_processed_json(path):
    """
    Load the preprocessed dataset from Part 1
    Each document is expected to be a JSON object containing
    text fields already cleaned and tokenized
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_index(docs):
    """
    Implement the inverted index

    Argument:
    docs -- collection of preprocessed documents

    Returns:
    index - the inverted index containing terms as keys and lists of document IDs as values
    pid_to_title - mapping of product IDs to titles (for result display)
    """
    index = defaultdict(set)  # Using set to avoid duplicate document IDs
    pid_to_title = {}

    for doc in docs:
        pid = doc.get("pid")
        if not pid:
            continue

        # Store title for result display
        pid_to_title[pid] = doc.get("title", f"product_{pid}")

        # Process all fields using tokenized text (with stopwords removed)
        terms = []
        for field in ("title_tokens", "description_tokens", "product_details_tokens"):
            if field in doc:
                terms.extend(doc[field])

        # Add document ID to each unstemmed term's posting list
        for term in set(terms):  # Using set to process each term once per document
            index[term].add(pid)

    # Convert sets to lists for JSON serialization
    return {term: list(docs) for term, docs in index.items()}, pid_to_title


def save_json(data, output_path):
    """
    Save a Python dictionary as a formatted JSON file
    
    Args:
        data (dict): Dictionary to be saved
        output_path (str): Destination file path
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}")


def compute_tfidf(index, docs):
    """
    Compute TF-IDF weights for each term in each document.

    Args:
        index (dict): Inverted index mapping terms to postings.
        docs (list[dict]): List of preprocessed product documents.

    Returns:
        tfidf (dict): {doc_id: {term: tfidf_value}}
    """
    N = len(docs)
    df = {term: len(postings) for term, postings in index.items()}
    tfidf = {}

    for doc in docs:
        pid = doc.get("pid")
        if not pid:
            continue

        tokens = []
        for field in ("title_tokens", "description_tokens", "product_details_tokens"):
            if field in doc:
                tokens.extend(doc[field])

        term_counts = Counter(tokens)
        norm = math.sqrt(sum(count ** 2 for count in term_counts.values()))
        tfidf[pid] = {}

        for term, count in term_counts.items():
            tf = count / norm if norm > 0 else 0
            idf = math.log((N + 1) / (df.get(term, 1) + 1)) + 1
            tfidf[pid][term] = tf * idf

    return tfidf




def rank_documents(query, tfidf, index):
    """
    Rank documents for a given query using conjunctive (AND) matching and TF-IDF weights.

    Args:
        query (str): Query text.
        tfidf (dict): TF-IDF representation of the corpus.
        index (dict): Inverted index for lookup.

    Returns:
        List of (doc_id, score) tuples sorted by descending score, only including
        documents that contain ALL query terms.
    """
    # First clean and tokenize the query (this includes stemming)
    cleaned = _clean_text(query)
    query_terms = _tokenize_and_normalize(cleaned)
    
    # Find documents containing all query terms (AND operation)
    matching_docs = set()
    first = True
    for term in query_terms:
        if term not in index:
            return []  # If any term is missing, no documents match
        
        # Get document IDs for this term
        docs = set(index[term])
        if first:
            matching_docs = docs
            first = False
        else:
            matching_docs &= docs  # Intersection for AND semantics
    
    # Calculate scores only for documents containing all terms
    scores = defaultdict(float)
    for pid in matching_docs:
        for term in query_terms:
            if term in tfidf.get(pid, {}):
                scores[pid] += tfidf[pid][term]

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs
