import sys
import json
from pathlib import Path
from tabulate import tabulate

# Add parent directories to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "project_progress" / "part_3"))
sys.path.insert(0, str(ROOT / "project_progress" / "part_2"))

from ranking_algorithms import compute_word2vec_model, rank_word2vec_cosine
from indexing import load_processed_json, create_index

TEST_QUERIES = [
    "women cotton sweatshirt",
    "men blue jeans slim fit",
    "red dress party",
    "running shoes sports",
    "leather jacket black"
]


def main():
    print("\n" + "=" * 80)
    print("WORD2VEC + COSINE SIMILARITY RANKING")
    print("=" * 80 + "\n")
    
    # Load data
    data_dir = ROOT / "data"
    input_path = data_dir / "processed_fashion.json"
    print("Loading dataset...")
    docs = load_processed_json(input_path)
    print(f"Loaded {len(docs)} documents.\n")
    
    # Build inverted index for AND semantics
    print("Building inverted index...")
    index, pid_to_title = create_index(docs)
    print(f"  {len(index)} terms indexed\n")
    
    # Train Word2Vec model
    print("Training Word2Vec model...")
    word2vec_model = compute_word2vec_model(docs)
    print(f"  Vocabulary size: {len(word2vec_model.wv)} words")
    print(f"  Vector dimension: {word2vec_model.vector_size}\n")
    
    # Process each query
    all_results = {}
    for query in TEST_QUERIES:
        print("=" * 80)
        print(f"QUERY: '{query}'")
        print("=" * 80 + "\n")
        
        results = rank_word2vec_cosine(query, word2vec_model, docs, index)
        all_results[query] = results
        
        if not results:
            print("⚠️  NO RESULTS (AND semantics - all query terms must appear in document)\n")
            continue
        
        # Display top-20
        top_20 = results[:20]
        table_data = []
        for rank, (pid, score) in enumerate(top_20, 1):
            title = pid_to_title.get(pid, "Unknown")
            table_data.append([rank, pid, title[:60], f"{score:.4f}"])
        
        headers = ["Rank", "Product ID", "Title", "Similarity"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print(f"\nTotal results: {len(results)} documents")
        print(f"Showing top-{len(top_20)} results\n")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80 + "\n")
    
    summary_data = []
    for query in TEST_QUERIES:
        results = all_results[query]
        summary_data.append([query, len(results)])
    
    print(tabulate(summary_data, headers=["Query", "Total Results"], tablefmt="grid"))
    
    print("\n" + "=" * 80)
    print("HOW WORD2VEC RANKING WORKS")
    print("=" * 80 + "\n")
    print("""
1. TRAINING:
   - Trains Word2Vec model on all product documents (title + description + details)
   - Creates dense vector embeddings for each word (100 dimensions)
   - Captures semantic relationships between words

2. QUERY PROCESSING:
   - Converts query terms to word vectors
   - Averages them to create query vector: (v1 + v2 + ... + vn) ÷ n

3. DOCUMENT SCORING:
   - For each document matching ALL query terms (AND semantics):
     * Converts document words to vectors
     * Averages them to create document vector
     * Computes cosine similarity with query vector
   - Ranks by similarity score (higher = more similar)

4. ADVANTAGES:
   ✓ Captures semantic similarity (not just exact matches)
   ✓ Handles synonyms and related terms
   ✓ Learns from corpus context

5. LIMITATIONS:
   ✗ Requires training data
   ✗ Averaging can lose word order information
   ✗ Still enforces AND semantics (all terms must appear)
    """)


if __name__ == "__main__":
    main()
