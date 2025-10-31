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
from array import array


# Test Queries for the exercise 2
TEST_QUERIES = [
    "women full sleeve sweatshirt cotton",
    "men slim fit jeans blue",
    "wireless bluetooth headphones",
    "leather wallet brown",
    "running shoes lightweight"
]



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
    lines -- collection of Wikipedia articles

    Returns:
    index - the inverted index (implemented through a Python dictionary) containing terms as keys and the corresponding
    list of documents where these keys appears in (and the positions) as values.
    """
    index = defaultdict(list)
    pid_to_title = {}  # dictionary to map page titles to page ids

    for doc in docs:  # Remember, lines contain all documents from file
        pid = doc.get("pid")
        if not pid:
            continue

        # Combinar tokens de todos los campos relevantes (title + description + details)
        tokens = []
        for field in ("title_tokens", "description_tokens", "product_details_tokens"):
            if field in doc:
                tokens.extend(doc[field])

        # Guardar el título asociado al pid
        pid_to_title[pid] = doc.get("title", f"product_{pid}")

        # Índice temporal para el documento actual
        current_doc_index = {}

        # Enumerar posiciones de términos
        for pos, term in enumerate(tokens):
            if term in current_doc_index:
                current_doc_index[term][1].append(pos)
            else:
                # 'I' = unsigned int array
                current_doc_index[term] = [pid, array('I', [pos])]

        # Fusionar el índice del documento con el índice global
        for term, posting in current_doc_index.items():
            index[term].append(posting)

    return index, pid_to_title






def build_vocabulary(index):
    """
    Build the vocabulary data structure from the inverted index

    Args: index(dict): the inverted index

    Returns:
        Vocabulary(dict): Contains:
            - term_id: numeric identifier
            - df: document frequency (number of docs containing the term)
            - cf: collection frequency (total term ocurrences)
    
    """
    vocabulary = {}
    for i, (term, postings) in enumerate(index.items(), start=1):
        df = len(postings)
        cf = sum(len(p[1]) for p in postings)
        vocabulary[term] = {"term_id":i, "df":df, "cf": cf}

    return vocabulary



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
    Rank documents for a given query using cosine similarity on TF-IDF weights.

    Args:
        query (str): Query text.
        tfidf (dict): TF-IDF representation of the corpus.
        index (dict): Inverted index for lookup.

    Returns:
        List of (doc_id, score) tuples sorted by descending score.
    """
    query_terms = query.lower().split()
    scores = defaultdict(float)

    for term in query_terms:
        postings = index.get(term, [])
        for pid, _ in postings:
            if term in tfidf.get(pid, {}):
                scores[pid] += tfidf[pid][term]

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs




def main():
    """
    Build the inverted index and vocabulary,
    then rank predefined test queries using the TF-IDF model.
    """
    input_path = "data/processed_fashion.json"
    output_index_path = "data/inverted_index.json"
    output_vocab_path = "data/vocabulary.json"

    # Load preprocessed dataset
    docs = load_processed_json(input_path)
    print(f"Loaded {len(docs)} processed documents.\n")

    # Build inverted index and vocabulary
    index, pid_to_title = create_index(docs)
    print(f"Inverted index built with {len(index)} unique terms.\n")
    vocabulary = build_vocabulary(index)
    print(f"Vocabulary created with {len(vocabulary)} entries.\n")

    # Save index and vocabulary
    save_json(index, output_index_path)
    save_json(vocabulary, output_vocab_path)

    # Compute TF-IDF
    print("Computing TF-IDF weights...")
    tfidf = compute_tfidf(index, docs)
    print("TF-IDF model computed.\n")

    # Rank predefined test queries
    print("Ranking sample queries using TF-IDF:\n")
    for q in TEST_QUERIES:
        ranked = rank_documents(q, tfidf, index)
        print(f"Query: '{q}'")
        for pid, score in ranked[:5]:
            title = pid_to_title.get(pid, "Unknown title")
            print(f" - {title} (score={score:.4f})")
        print()





# Entry point

if __name__ == "__main__":
    main()
