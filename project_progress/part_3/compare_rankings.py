import sys
import json
from pathlib import Path
from tabulate import tabulate

# Add parent directories to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "project_progress" / "part_3"))
sys.path.insert(0, str(ROOT / "project_progress" / "part_2"))

from ranking_algorithms import (
    compute_tfidf_vectors,
    rank_tfidf_cosine,
    compute_bm25_index,
    rank_bm25,
    compute_custom_index,
    rank_custom_score,
)
from indexing import load_processed_json, create_index

TEST_QUERIES = [
    "women cotton sweatshirt",
    "men blue jeans slim fit",
    "red dress party",
    "running shoes sports",
    "leather jacket black"
]


def load_data():
    data_dir = ROOT / "data"
    input_path = data_dir / "processed_fashion.json"
    print("Loading dataset...")
    docs = load_processed_json(input_path)
    print(f"Loaded {len(docs)} documents.\n")
    return docs


def build_indices(docs):
    print("Building indices...")
    
    index, pid_to_title = create_index(docs)
    print(f"  Inverted index: {len(index)} terms")
    
    tfidf_vectors = compute_tfidf_vectors(index, docs)
    print(f"  TF-IDF vectors: {len(tfidf_vectors)} docs")
    
    bm25_index, doc_lengths, avg_doc_length = compute_bm25_index(docs)
    print(f"  BM25 index: avg length {avg_doc_length:.1f}")
    
    doc_metadata = compute_custom_index(docs)
    print(f"  Metadata: {len(doc_metadata)} docs\n")
    
    return index, pid_to_title, tfidf_vectors, bm25_index, doc_lengths, avg_doc_length, doc_metadata


def compare_query_results(query, index, pid_to_title, tfidf_vectors, bm25_index, 
                          doc_lengths, avg_doc_length, doc_metadata, num_docs, top_k=10):
    print("=" * 100)
    print(f"QUERY: '{query}'")
    print("=" * 100)
    
    results_tfidf = rank_tfidf_cosine(query, tfidf_vectors, index, num_docs)
    results_bm25 = rank_bm25(query, bm25_index, doc_lengths, avg_doc_length, num_docs)
    results_custom = rank_custom_score(query, tfidf_vectors, index, num_docs, doc_metadata)
    
    if not results_tfidf and not results_bm25 and not results_custom:
        print("\n⚠️  NO RESULTS (AND semantics)\n")
        return
    
    print(f"\nTop {top_k} Results:\n")
    
    max_rows = min(max(len(results_tfidf), len(results_bm25), len(results_custom)), top_k)
    
    table_data = []
    for i in range(max_rows):
        row = [f"{i+1}"]
        
        if i < len(results_tfidf):
            pid, score = results_tfidf[i]
            row.extend([pid_to_title.get(pid, "Unknown")[:40], f"{score:.4f}"])
        else:
            row.extend(["—", "—"])
        
        if i < len(results_bm25):
            pid, score = results_bm25[i]
            row.extend([pid_to_title.get(pid, "Unknown")[:40], f"{score:.4f}"])
        else:
            row.extend(["—", "—"])
        
        if i < len(results_custom):
            pid, score = results_custom[i]
            row.extend([pid_to_title.get(pid, "Unknown")[:40], f"{score:.4f}"])
        else:
            row.extend(["—", "—"])
        
        table_data.append(row)
    
    headers = ["Rank", "TF-IDF+Cosine", "Score", "BM25", "Score", "Custom", "Score"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print(f"\nResults: TF-IDF={len(results_tfidf)}, BM25={len(results_bm25)}, Custom={len(results_custom)}\n")


def analyze_ranking_differences(query, index, pid_to_title, tfidf_vectors, bm25_index, 
                                doc_lengths, avg_doc_length, doc_metadata, num_docs, top_k=5):
    results_tfidf = rank_tfidf_cosine(query, tfidf_vectors, index, num_docs)
    results_bm25 = rank_bm25(query, bm25_index, doc_lengths, avg_doc_length, num_docs)
    results_custom = rank_custom_score(query, tfidf_vectors, index, num_docs, doc_metadata)
    
    if not results_tfidf:
        return
    
    top_tfidf = set(pid for pid, _ in results_tfidf[:top_k])
    top_bm25 = set(pid for pid, _ in results_bm25[:top_k])
    top_custom = set(pid for pid, _ in results_custom[:top_k])
    
    overlap_tfidf_bm25 = len(top_tfidf & top_bm25)
    overlap_tfidf_custom = len(top_tfidf & top_custom)
    overlap_bm25_custom = len(top_bm25 & top_custom)
    overlap_all = len(top_tfidf & top_bm25 & top_custom)
    
    print(f"  Ranking Overlap Analysis (Top {top_k}):")
    print(f"  TF-IDF ∩ BM25:    {overlap_tfidf_bm25}/{top_k} documents")
    print(f"  TF-IDF ∩ Custom:  {overlap_tfidf_custom}/{top_k} documents")
    print(f"  BM25 ∩ Custom:    {overlap_bm25_custom}/{top_k} documents")
    print(f"  All Three:        {overlap_all}/{top_k} documents")
    print()


def main():
    print("\n" + "=" * 100)
    print("PART 3: RANKING ALGORITHMS COMPARISON")
    print("=" * 100 + "\n")
    
    docs = load_data()
    num_docs = len(docs)
    indices = build_indices(docs)
    index, pid_to_title, tfidf_vectors, bm25_index, doc_lengths, avg_doc_length, doc_metadata = indices
    
    for query in TEST_QUERIES:
        compare_query_results(
            query, index, pid_to_title, tfidf_vectors, 
            bm25_index, doc_lengths, avg_doc_length, doc_metadata, num_docs
        )
        analyze_ranking_differences(
            query, index, pid_to_title, tfidf_vectors, 
            bm25_index, doc_lengths, avg_doc_length, doc_metadata, num_docs
        )
    
    print("\n" + "=" * 100)
    print("ANALYSIS SUMMARY")
    print("=" * 100 + "\n")
    
    print("""
   Key Observations:

1. TF-IDF + Cosine Similarity:
   ✓ Pros: Simple, interpretable, good baseline
   ✓ Cons: Treats all term occurrences equally (no saturation)
   ✓ Best for: Queries where exact term matching is important

2. BM25:
   ✓ Pros: Term frequency saturation, length normalization, industry standard
   ✓ Cons: More complex, requires parameter tuning
   ✓ Best for: Production systems, handles repeated terms well

3. Custom Score (Text + Quality + Value):
   ✓ Pros: Domain-specific, incorporates product features, user-centric
   ✓ Cons: More complex, requires weight tuning, domain-specific
   ✓ Best for: E-commerce where product quality matters

The rankings differ because:
- TF-IDF focuses purely on text similarity
- BM25 normalizes for document length and term saturation
- Custom Score balances text relevance with product quality (ratings, discounts)
    """)


if __name__ == "__main__":
    main()
