"""
Test script for indexing functions
Builds the inverted index and vocabulary, then ranks predefined test queries using TF-IDF
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "project_progress" / "part_2"))

from indexing import (
    load_processed_json,
    create_index,
    save_json,
    compute_tfidf,
    rank_documents
)


# Test Queries for the exercise 2
TEST_QUERIES = [
    "cotton blue jeans",          # Testing common clothing terms + color
    "women track pants",          # Testing gender + clothing category
    "multicolor dress casual",    # Testing color + category + style
    "cotton gucci pants",         # Testing AND with non-existent brand term (Should give No result)
    "cotton silk blend"           # Testing multiple fabric combinations
]


def main():
    """
    Build the inverted index and vocabulary,
    then rank predefined test queries using the TF-IDF model.
    """
    data_dir = ROOT / "data"
    input_path = data_dir / "processed_fashion.json"
    output_index_path = data_dir / "inverted_index.json"

    # Load preprocessed dataset
    docs = load_processed_json(input_path)
    print(f"Loaded {len(docs)} processed documents.\n")

    # Build inverted index
    index, pid_to_title = create_index(docs)
    print(f"Inverted index built with {len(index)} unique terms.\n")

    # Save index
    save_json(index, output_index_path)

    # Compute TF-IDF
    print("Computing TF-IDF weights...")
    tfidf = compute_tfidf(index, docs)
    print("TF-IDF model computed.\n")

    # Rank predefined test queries
    print("\nRanking sample queries using TF-IDF:\n")
    for q in TEST_QUERIES:
        print(f"Query: '{q}'")
        ranked = rank_documents(q, tfidf, index)
        if not ranked:
            print(" No results found - all query terms must be present in document (AND operation)")
        else:
            for pid, score in ranked[:5]:
                title = pid_to_title.get(pid, "Unknown title")
                print(f" - {title} (score={score:.4f})")
        print()


if __name__ == "__main__":
    main()
