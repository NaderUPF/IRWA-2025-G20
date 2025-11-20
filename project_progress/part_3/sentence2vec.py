import sys
import json
from pathlib import Path
from tabulate import tabulate
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directories to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "project_progress" / "part_3"))
sys.path.insert(0, str(ROOT / "project_progress" / "part_2"))

from ranking_algorithms import compute_word2vec_model
from indexing import load_processed_json, create_index

TEST_QUERIES = [
    "women cotton sweatshirt",
    "men blue jeans slim fit",
    "red dress party",
    "running shoes sports",
    "leather jacket black"
]

def sentence2vec(tokens, model):
    """Compute document embedding as mean of word embeddings."""
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def rank_sentence2vec(query, model, docs, index):
    """Ranking identical to Word2Vec pipeline, but vector representation = Sentence2Vec."""

    from project_progress.part_1.preprocess import _clean_text, _tokenize_and_normalize

    cleaned = _clean_text(query)
    query_tokens = _tokenize_and_normalize(cleaned)


    # AND semantics using inverted index (SAME AS WORD2VEC)
    posting_lists = []
    for term in query_tokens:
        if term not in index:
            return []  # if a word doesn't exist → zero docs
        posting_lists.append(set(index[term]))

    matching_pids = set.intersection(*posting_lists)

    # create embedding for query
    query_vec = sentence2vec(query_tokens, model)

    results = []

    # compute similarity only on AND-filtered docs
    for doc in docs:
        pid = doc["pid"]
        if pid not in matching_pids:
            continue

        # use same token fields as Word2Vec, but averaged as sentence representation
        tokens = []
        tokens += doc.get("title_tokens", [])
        tokens += doc.get("description_tokens", [])
        tokens += doc.get("product_details_tokens", [])

        doc_vec = sentence2vec(tokens, model)
        score = cosine_similarity([query_vec], [doc_vec])[0][0]

        results.append((pid, score))

    return sorted(results, key=lambda x: x[1], reverse=True)


# --------------------------
# MAIN
# --------------------------

def main():
    print("\n" + "=" * 80)
    print("SENTENCE2VEC + COSINE SIMILARITY RANKING")
    print("=" * 80 + "\n")

    # Load dataset
    data_dir = ROOT / "data"
    input_path = data_dir / "processed_fashion.json"
    print("Loading dataset...")
    docs = load_processed_json(input_path)
    print(f"Loaded {len(docs)} documents.\n")

    # Build index (AND semantics)
    print("Building inverted index...")
    index, pid_to_title = create_index(docs)
    print(f"  {len(index)} terms indexed\n")

    # Train embeddings (same model as Word2Vec)
    print("Training Word2Vec model...")
    word2vec_model = compute_word2vec_model(docs)
    print(f"  Vocabulary size: {len(word2vec_model.wv)} words")
    print(f"  Vector dimension: {word2vec_model.vector_size}\n")

    # Run queries
    all_results = {}
    for query in TEST_QUERIES:
        print("=" * 80)
        print(f"QUERY: '{query}'")
        print("=" * 80 + "\n")

        results = rank_sentence2vec(query, word2vec_model, docs, index)
        all_results[query] = results

        if not results:
            print("⚠️  NO RESULTS (AND semantics - all query terms must appear)\n")
            continue

        # Show top-20
        top_20 = results[:20]
        table_data = []
        for rank, (pid, score) in enumerate(top_20, 1):
            title = pid_to_title.get(pid, "Unknown")
            table_data.append([rank, pid, title[:60], f"{score:.4f}"])

        headers = ["Rank", "Product ID", "Title", "Similarity"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        print(f"\nTotal results: {len(results)}")
        print(f"Showing top-{len(top_20)} results\n")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80 + "\n")

    summary_data = []
    for query in TEST_QUERIES:
        summary_data.append([query, len(all_results[query])])

    print(tabulate(summary_data, headers=["Query", "Total Results"], tablefmt="grid"))

    print("\n" + "=" * 80)
    print("HOW SENTENCE2VEC RANKING WORKS")
    print("=" * 80 + "\n")
    print("""
1. TRAINING:
   - Trains Word2Vec model on all product documents
   - Sentence vector = average of all word vectors

2. QUERY PROCESSING:
   - Query tokens = lowercase split
   - Query vector = mean(word vectors)

3. DOCUMENT SCORING:
   - AND semantics via inverted index (same as Word2Vec)
   - Cosine similarity between query vector and doc vector

4. DIFFERENCE FROM WORD2VEC:
   - Representation is full-document average, not just context windows
   - Ranking changes, filtering stays identical
    """)


if __name__ == "__main__":
    main()