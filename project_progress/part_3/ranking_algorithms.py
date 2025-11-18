import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from project_progress.part_1.preprocess import _clean_text, _tokenize_and_normalize


# ============================================================================
# 1. TF-IDF with Cosine Similarity (CORRECTED)
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

def compute_bm25_index(docs: List[Dict]) -> Tuple[Dict, Dict, float]:
    index = defaultdict(list)
    doc_lengths = {}
    total_length = 0
    
    for doc in docs:
        pid = doc.get("pid")
        if not pid:
            continue
        
        tokens = []
        for field in ("title_tokens", "description_tokens", "product_details_tokens"):
            if field in doc:
                tokens.extend(doc[field])
        
        doc_lengths[pid] = len(tokens)
        total_length += len(tokens)
        
        term_freqs = Counter(tokens)
        for term, freq in term_freqs.items():
            index[term].append((pid, freq))
    
    avg_doc_length = total_length / len(docs) if docs else 0
    return dict(index), doc_lengths, avg_doc_length


def rank_bm25(query: str, bm25_index: Dict, doc_lengths: Dict, avg_doc_length: float, 
              num_docs: int, k1: float = 1.5, b: float = 0.75) -> List[Tuple[str, float]]:
    # k1 controls TF saturation, b controls length normalization
    cleaned = _clean_text(query)
    query_terms = _tokenize_and_normalize(cleaned)
    if not query_terms:
        return []
    
    # AND semantics
    matching_docs = set()
    first = True
    for term in query_terms:
        if term not in bm25_index:
            return []
        docs = set(doc_id for doc_id, _ in bm25_index[term])
        if first:
            matching_docs = docs
            first = False
        else:
            matching_docs &= docs
    
    if not matching_docs:
        return []
    
    term_doc_freqs = {}
    for term in query_terms:
        term_doc_freqs[term] = {doc_id: freq for doc_id, freq in bm25_index[term]}
    
    scores = defaultdict(float)
    for term in query_terms:
        df = len(bm25_index[term])
        idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
        
        for doc_id in matching_docs:
            if doc_id in term_doc_freqs[term]:
                freq = term_doc_freqs[term][doc_id]
                doc_len = doc_lengths.get(doc_id, 0)
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * (doc_len / avg_doc_length if avg_doc_length > 0 else 1))
                scores[doc_id] += idf * (numerator / denominator)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


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
