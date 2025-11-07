"""
Evaluation metrics for Information Retrieval systems.
Implements various ranking-based evaluation metrics.
"""

import numpy as np
from collections import defaultdict


def precision_at_k(doc_score, y_score, k=10):
    """
    Compute Precision@K - proportion of relevant documents in top K results.
    
    Parameters
    ----------
    doc_score: Ground truth (binary relevance labels).
    y_score: Predicted relevance scores.
    k: Number of documents to consider.
    
    Returns
    -------
    precision@k : float
    """
    order = np.argsort(y_score)[::-1]  # Sort by descending score
    doc_score = doc_score[order[:k]]  # Get top K relevance labels
    return float(np.sum(doc_score == 1)) / k


def recall_at_k(doc_score, y_score, k=10):
    """
    Compute Recall@K - proportion of all relevant documents found in top K results.
    
    Parameters
    ----------
    doc_score: Ground truth (binary relevance labels).
    y_score: Predicted relevance scores.
    k: Number of documents to consider.
    
    Returns
    -------
    recall@k : float
    """
    order = np.argsort(y_score)[::-1]  # Sort by descending score
    doc_score_k = doc_score[order[:k]]  # Get top K relevance labels
    total_relevant = np.sum(doc_score == 1)  # Total number of relevant docs
    if total_relevant == 0:
        return 0.0
    return float(np.sum(doc_score_k == 1)) / total_relevant


def f1_score_at_k(doc_score, y_score, k=10):
    """
    Compute F1-Score@K - harmonic mean of precision and recall at K.
    
    Parameters
    ----------
    doc_score: Ground truth (binary relevance labels).
    y_score: Predicted relevance scores.
    k: Number of documents to consider.
    
    Returns
    -------
    f1@k : float
    """
    precision = precision_at_k(doc_score, y_score, k)
    recall = recall_at_k(doc_score, y_score, k)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def avg_precision_at_k(doc_score, y_score, k=10):
    """
    Compute Average Precision@K.
    
    Parameters
    ----------
    doc_score: Ground truth (binary relevance labels).
    y_score: Predicted relevance scores.
    k: Number of documents to consider.
    
    Returns
    -------
    average precision@k : float
    """
    order = np.argsort(y_score)[::-1]  # Sort by descending score
    
    prec_at_i = 0
    prec_list = []
    num_relevant = 0
    num_to_iterate = min(k, len(order))
    
    for i in range(num_to_iterate):
        if doc_score[order[i]] == 1:
            num_relevant += 1
            prec_at_i = num_relevant / (i + 1)
            prec_list.append(prec_at_i)
    
    if num_relevant == 0:
        return 0.0
    
    return np.sum(prec_list) / num_relevant


def mean_average_precision(search_results, k=10):
    """
    Compute Mean Average Precision across all queries.
    
    Parameters
    ----------
    search_results: DataFrame containing:
        - query_id: query identifier
        - doc_id: document identifier
        - predicted_relevance: predicted relevance scores
        - is_relevant: binary relevance labels
    k: Number of documents to consider per query.
    
    Returns
    -------
    map@k : float, average precisions : list
    """
    avg_precisions = []
    
    for q in search_results["query_id"].unique():
        curr_results = search_results[search_results["query_id"] == q]
        ap = avg_precision_at_k(
            np.array(curr_results["is_relevant"]),
            np.array(curr_results["predicted_relevance"]), 
            k
        )
        avg_precisions.append(ap)
    
    return np.mean(avg_precisions), avg_precisions


def reciprocal_rank_at_k(doc_score, y_score, k=10):
    """
    Compute Reciprocal Rank at K - inverse of rank of first relevant document.
    
    Parameters
    ----------
    doc_score: Ground truth (binary relevance labels).
    y_score: Predicted relevance scores.
    k: Maximum rank to consider.
    
    Returns
    -------
    rr@k : float
    """
    order = np.argsort(y_score)[::-1]  # Sort by descending score
    doc_score = doc_score[order[:k]]
    
    # Find position of first relevant document
    relevant_positions = np.where(doc_score == 1)[0]
    if len(relevant_positions) == 0:
        return 0.0
    
    rank = relevant_positions[0] + 1  # Convert to 1-based rank
    return 1.0 / rank if rank <= k else 0.0


def mean_reciprocal_rank(search_results, k=10):
    """
    Compute Mean Reciprocal Rank across all queries.
    
    Parameters
    ----------
    search_results: DataFrame containing search results
    k: Maximum rank to consider
    
    Returns
    -------
    mrr@k : float
    """
    rr_values = []
    
    for q in search_results["query_id"].unique():
        curr_results = search_results[search_results["query_id"] == q]
        rr = reciprocal_rank_at_k(
            np.array(curr_results["is_relevant"]),
            np.array(curr_results["predicted_relevance"]), 
            k
        )
        rr_values.append(rr)
    
    return np.mean(rr_values)


def dcg_at_k(doc_score, y_score, k=10):
    """
    Compute Discounted Cumulative Gain at K.
    Uses the formula: DCG = sum((2^rel - 1) / log2(pos + 1))
    
    Parameters
    ----------
    doc_score: Ground truth (relevance labels, can be graded).
    y_score: Predicted relevance scores.
    k: Number of documents to consider.
    
    Returns
    -------
    dcg@k : float
    """
    order = np.argsort(y_score)[::-1]  # Sort by descending score
    doc_score = doc_score[order[:k]]
    
    gains = 2 ** doc_score - 1
    discounts = np.log2(np.arange(len(doc_score)) + 2)
    return np.sum(gains / discounts)


def ndcg_at_k(doc_score, y_score, k=10):
    """
    Compute Normalized Discounted Cumulative Gain at K.
    
    Parameters
    ----------
    doc_score: Ground truth (relevance labels, can be graded).
    y_score: Predicted relevance scores.
    k: Number of documents to consider.
    
    Returns
    -------
    ndcg@k : float
    """
    # Compute ideal DCG (sort by true relevance)
    ideal_dcg = dcg_at_k(doc_score, doc_score, k)
    if ideal_dcg == 0:
        return 0.0
        
    # Compute actual DCG
    dcg = dcg_at_k(doc_score, y_score, k)
    return dcg / ideal_dcg


def evaluate_query(query_results, k_values=[5, 10]):
    """
    Evaluate a single query using multiple metrics and k values.
    
    Parameters
    ----------
    query_results: DataFrame containing results for one query
    k_values: List of k values to compute metrics for
    
    Returns
    -------
    metrics: dict of metric values
    """
    metrics = {}
    doc_scores = np.array(query_results["is_relevant"])
    y_scores = np.array(query_results["predicted_relevance"])
    
    for k in k_values:
        metrics[f'P@{k}'] = round(precision_at_k(doc_scores, y_scores, k), 3)
        metrics[f'R@{k}'] = round(recall_at_k(doc_scores, y_scores, k), 3)
        metrics[f'F1@{k}'] = round(f1_score_at_k(doc_scores, y_scores, k), 3)
        metrics[f'AP@{k}'] = round(avg_precision_at_k(doc_scores, y_scores, k), 3)
        metrics[f'RR@{k}'] = round(reciprocal_rank_at_k(doc_scores, y_scores, k), 3)
        metrics[f'NDCG@{k}'] = round(ndcg_at_k(doc_scores, y_scores, k), 3)
    
    return metrics