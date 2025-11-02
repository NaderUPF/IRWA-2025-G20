"""
Evaluation script for Information Retrieval System.

This script:
1. Loads the existing validation_labels.csv with ground truth
2. Runs the search engine on validation queries
3. Computes all evaluation metrics (P@K, R@K, F1@K, AP@K, MAP, MRR, NDCG@K)
4. Displays results for each query and overall performance
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directories to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "project_progress" / "part_2"))

from evaluation import evaluate_query, mean_average_precision, mean_reciprocal_rank
from indexing import load_processed_json, create_index, compute_tfidf, rank_documents


# Validation queries from the assignment
VALIDATION_QUERIES = {
    1: "women full sleeve sweatshirt cotton",
    2: "men slim jeans blue"
}


def load_validation_labels():
    """Load validation labels from CSV"""
    data_dir = ROOT / "data"
    labels_df = pd.read_csv(data_dir / "validation_labels.csv")
    return labels_df


def run_search_and_get_scores(query, index, tfidf, pid_to_title):
    """
    Run search for a query and return results with scores
    
    Returns:
    --------
    DataFrame with columns: pid, title, predicted_relevance
    """
    # Get ranked results
    ranked = rank_documents(query, tfidf, index)
    
    # Create results dataframe
    results = []
    for pid, score in ranked:
        results.append({
            'pid': pid,
            'title': pid_to_title.get(pid, 'Unknown'),
            'predicted_relevance': score
        })
    
    return pd.DataFrame(results)


def prepare_evaluation_data(query_id, search_results_df, labels_df):
    """
    Merge search results with ground truth labels for evaluation
    
    Returns:
    --------
    DataFrame with columns: query_id, pid, title, predicted_relevance, is_relevant
    """
    # Get ground truth for this query
    query_labels = labels_df[labels_df['query_id'] == query_id][['pid', 'labels']].copy()
    query_labels = query_labels.rename(columns={'labels': 'is_relevant'})
    
    # Merge with search results
    eval_df = search_results_df.merge(query_labels, on='pid', how='left')
    
    # Fill missing labels with 0 (not relevant)
    eval_df['is_relevant'] = eval_df['is_relevant'].fillna(0).astype(int)
    eval_df['query_id'] = query_id
    
    return eval_df


def print_top_results(query_results, k=10):
    """Print top K results with relevance labels"""
    top_k = query_results.head(k)
    print(f"\n   Top {k} Results:")
    for i, row in enumerate(top_k.itertuples(), 1):
        marker = "✓" if row.is_relevant == 1 else "✗"
        print(f"   {i:2d}. [{marker}] {row.title[:60]}")
        print(f"       Score: {row.predicted_relevance:.4f}")


def main():
    print("\n" + "=" * 80)
    print(" " * 20 + "SEARCH SYSTEM EVALUATION")
    print("=" * 80)
    
    # Step 1: Load data and build index
    print("\n[1] Loading data and building search index...")
    data_dir = ROOT / "data"
    docs = load_processed_json(data_dir / "processed_fashion.json")
    index, pid_to_title = create_index(docs)
    tfidf = compute_tfidf(index, docs)
    print(f"Loaded {len(docs)} documents")
    print(f"Index contains {len(index)} unique terms")
    
    # Step 2: Load validation labels
    print("\n[2] Loading ground truth labels...")
    labels_df = load_validation_labels()
    print(f"Loaded {len(labels_df)} labeled document-query pairs")
    print(f"Queries: {labels_df['query_id'].nunique()}")
    
    # Step 3: Evaluate each query
    print("\n[3] Running evaluation...")
    print("=" * 80)
    
    all_eval_data = []
    k_values = [5, 10]
    
    for query_id, query_text in VALIDATION_QUERIES.items():
        print(f"\n   Query {query_id}: '{query_text}'")
        print("-" * 80)
        
        # Run search
        search_results = run_search_and_get_scores(query_text, index, tfidf, pid_to_title)
        
        # Prepare evaluation data
        eval_data = prepare_evaluation_data(query_id, search_results, labels_df)
        all_eval_data.append(eval_data)
        
        # Show top results
        print_top_results(eval_data, k=10)
        
        # Calculate metrics
        metrics = evaluate_query(eval_data, k_values)
        
        # Print metrics in clean format
        print(f"\n   Evaluation Metrics:")
        print(f"   {'-'*40}")
        for k in k_values:
            print(f"\n   @ K={k}:")
            print(f"      Precision@{k}  : {metrics[f'P@{k}']:.3f}")
            print(f"      Recall@{k}     : {metrics[f'R@{k}']:.3f}")
            print(f"      F1-Score@{k}   : {metrics[f'F1@{k}']:.3f}")
            print(f"      AP@{k}         : {metrics[f'AP@{k}']:.3f}")
            print(f"      RR@{k}         : {metrics[f'RR@{k}']:.3f}")
            print(f"      NDCG@{k}       : {metrics[f'NDCG@{k}']:.3f}")
    
    # Step 4: Calculate overall metrics
    print("\n" + "=" * 80)
    print(" " * 25 + "OVERALL METRICS")
    print("=" * 80)
    
    all_eval_df = pd.concat(all_eval_data, ignore_index=True)
    
    for k in k_values:
        map_k, _ = mean_average_precision(all_eval_df, k=k)
        mrr_k = mean_reciprocal_rank(all_eval_df, k=k)
        print(f"\n@ K={k}:")
        print(f"   Mean Average Precision (MAP@{k}) : {map_k:.3f}")
        print(f"   Mean Reciprocal Rank (MRR@{k})   : {mrr_k:.3f}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_queries = len(VALIDATION_QUERIES)
    total_results = len(all_eval_df)
    total_relevant = all_eval_df['is_relevant'].sum()
    
    print(f"Total Queries Evaluated    : {total_queries}")
    print(f"Total Results Returned     : {total_results}")
    print(f"Total Relevant Documents   : {total_relevant}")
    print(f"Average Results per Query  : {total_results / total_queries:.1f}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
