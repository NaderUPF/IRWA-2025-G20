import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from rank_bm25 import BM25Okapi
import numpy as np
from gensim.models import Word2Vec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from project_progress.part_1.preprocess import _clean_text, _tokenize_and_normalize


# ============================================================================
# 1. TF-IDF with Cosine Similarity (CORRECTED FROM PART_2)
# ============================================================================

def compute_tfidf_vectors(index: Dict, docs: List[Dict]) -> Dict[str, Dict[str, float]]:
    N = len(docs)
    df = {term: len(postings) for term, postings in index.items()}
    tfidf_vectors = {}
    
    for doc in docs:
        pid = doc.get("pid")
        if not pid:
            continue
        
        tokens = []
        for field in ("title_tokens", "description_tokens", "product_details_tokens"):
            if field in doc:
                tokens.extend(doc[field])
        
        term_counts = Counter(tokens)
        total_terms = len(tokens)
        
        tfidf_vectors[pid] = {}
        for term, count in term_counts.items():
            tf = count / total_terms if total_terms > 0 else 0
            idf = math.log((N + 1) / (df.get(term, 0) + 1)) + 1
            tfidf_vectors[pid][term] = tf * idf
    
    return tfidf_vectors


def compute_query_tfidf(query_terms: List[str], index: Dict, num_docs: int) -> Dict[str, float]:
    query_vector = {}
    term_counts = Counter(query_terms)
    total_terms = len(query_terms)
    
    for term, count in term_counts.items():
        tf = count / total_terms if total_terms > 0 else 0
        df = len(index.get(term, []))
        idf = math.log((num_docs + 1) / (df + 1)) + 1
        query_vector[term] = tf * idf
    
    return query_vector


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in set(vec1) | set(vec2))
    mag1 = math.sqrt(sum(w ** 2 for w in vec1.values()))
    mag2 = math.sqrt(sum(w ** 2 for w in vec2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)


def rank_tfidf_cosine(query: str, tfidf_vectors: Dict, index: Dict, num_docs: int) -> List[Tuple[str, float]]:
    cleaned = _clean_text(query)
    query_terms = _tokenize_and_normalize(cleaned)
    if not query_terms:
        return []
    
    # AND semantics: docs must contain ALL query terms
    matching_docs = set()
    first = True
    for term in query_terms:
        if term not in index:
            return []
        docs = set(index[term])
        if first:
            matching_docs = docs
            first = False
        else:
            matching_docs &= docs
    
    if not matching_docs:
        return []
    
    query_vector = compute_query_tfidf(query_terms, index, num_docs)
    scores = {}
    for doc_id in matching_docs:
        doc_vector = tfidf_vectors.get(doc_id, {})
        scores[doc_id] = cosine_similarity(query_vector, doc_vector)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ============================================================================
# 2. BM25
# ============================================================================

def compute_bm25_index(docs: List[Dict]) -> Tuple[BM25Okapi, Dict]:
    """Build BM25 index and keep track of doc IDs"""
    doc_tokens_list = []
    pid_list = []
    
    for doc in docs:
        pid = doc.get("pid")
        if not pid:
            continue
        
        tokens = []
        for field in ("title_tokens", "description_tokens", "product_details_tokens"):
            if field in doc:
                tokens.extend(doc[field])
        
        doc_tokens_list.append(tokens)
        pid_list.append(pid)
    
    bm25 = BM25Okapi(doc_tokens_list)
    pid_to_index = {pid: i for i, pid in enumerate(pid_list)}
    
    return bm25, pid_to_index, pid_list


def rank_bm25(query: str, bm25: BM25Okapi, pid_to_index: Dict, pid_list: List[str]) -> List[Tuple[str, float]]:
    cleaned = _clean_text(query)
    query_terms = _tokenize_and_normalize(cleaned)
    if not query_terms:
        return []
    
    # Get BM25 scores for all documents
    scores = bm25.get_scores(query_terms)
    
    # Create list of (pid, score) tuples
    results = [(pid_list[i], scores[i]) for i in range(len(pid_list))]
    
    # Filter for AND semantics: only keep docs with non-zero scores
    # (BM25 returns 0 for docs missing query terms)
    results = [(pid, score) for pid, score in results if score > 0]
    
    return sorted(results, key=lambda x: x[1], reverse=True)


# ============================================================================
# 4. Word2Vec + Cosine Similarity
# ============================================================================

def compute_word2vec_model(docs: List[Dict], vector_size: int = 100, 
                          window: int = 5, min_count: int = 1) -> Word2Vec:
    sentences = []
    for doc in docs:
        # Combine all token fields
        tokens = []
        for field in ("title_tokens", "description_tokens", "product_details_tokens"):
            if field in doc:
                tokens.extend(doc[field])
        
        if tokens:
            sentences.append(tokens)
    
    if not sentences:
        # Return empty model if no sentences
        model = Word2Vec(min_count=min_count, vector_size=vector_size)
        return model
    
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, 
                     min_count=min_count, workers=4, sg=0, epochs=5)
    return model


def text_to_vector(tokens: List[str], model: Word2Vec) -> np.ndarray:
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    
    if not vectors:
        return np.zeros(model.vector_size)
    
    return np.mean(vectors, axis=0)


def rank_word2vec_cosine(query: str, model: Word2Vec, docs: List[Dict], 
                         index: Dict) -> List[Tuple[str, float]]:
    cleaned = _clean_text(query)
    query_terms = _tokenize_and_normalize(cleaned)
    if not query_terms:
        return []
    
    # AND semantics: get candidate docs containing all query terms
    query_set = set(query_terms)
    candidate_pids = None
    for term in query_set:
        term_pids = set(index.get(term, []))
        if candidate_pids is None:
            candidate_pids = term_pids
        else:
            candidate_pids &= term_pids
    
    if not candidate_pids:
        return []
    
    # Compute query vector
    query_vec = text_to_vector(query_terms, model)
    if np.linalg.norm(query_vec) == 0:
        return []
    
    # Create doc lookup by pid
    doc_by_pid = {doc.get("pid"): doc for doc in docs if doc.get("pid")}
    
    # Compute cosine similarity for each candidate doc
    results = []
    for pid in candidate_pids:
        doc = doc_by_pid.get(pid)
        if not doc:
            continue
        
        # Use same fields as BM25
        doc_tokens = []
        for field in ("title_tokens", "description_tokens", "product_details_tokens"):
            if field in doc:
                doc_tokens.extend(doc[field])
        
        if not doc_tokens:
            continue
        
        doc_vec = text_to_vector(doc_tokens, model)
        if np.linalg.norm(doc_vec) == 0:
            continue
        
        # Cosine similarity
        similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
        results.append((pid, similarity))
    
    return sorted(results, key=lambda x: x[1], reverse=True)


# ============================================================================
# 3. Custom Score (Text + Quality + Value)
# ============================================================================

def compute_custom_index(docs: List[Dict]) -> Dict:
    doc_metadata = {}
    for doc in docs:
        pid = doc.get("pid")
        if not pid:
            continue
        doc_metadata[pid] = {
            "average_rating": doc.get("average_rating", 0.0) or 0.0,
            "discount": doc.get("discount", 0.0) or 0.0,
            "selling_price": doc.get("selling_price", 0.0) or 0.0,
            "out_of_stock": doc.get("out_of_stock", False),
            "brand": doc.get("brand", "").lower(),
        }
    return doc_metadata


def rank_custom_score(query: str, tfidf_vectors: Dict, index: Dict, num_docs: int, 
                      doc_metadata: Dict, alpha: float = 0.6, beta: float = 0.25, 
                      gamma: float = 0.15) -> List[Tuple[str, float]]:
    # Custom score = 60% text + 25% quality + 15% value
    tfidf_results = rank_tfidf_cosine(query, tfidf_vectors, index, num_docs)
    if not tfidf_results:
        return []
    
    # Normalize text scores to [0,1]
    max_tfidf = max(score for _, score in tfidf_results)
    tfidf_scores = {doc_id: score / max_tfidf for doc_id, score in tfidf_results}
    
    final_scores = {}
    for doc_id, text_score in tfidf_scores.items():
        metadata = doc_metadata.get(doc_id, {})
        
        # Quality: rating + stock availability
        rating = metadata.get("average_rating", 0.0)
        rating_normalized = rating / 5.0
        stock_penalty = 0.5 if metadata.get("out_of_stock", False) else 1.0
        quality_score = rating_normalized * stock_penalty
        
        # Value: discount
        discount = metadata.get("discount", 0.0)
        value_score = min(discount / 100.0, 1.0)
        
        final_score = alpha * text_score + beta * quality_score + gamma * value_score
        final_scores[doc_id] = final_score
    
    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
