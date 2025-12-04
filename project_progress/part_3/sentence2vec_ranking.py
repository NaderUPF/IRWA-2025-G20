import sys
import json
from pathlib import Path
from tabulate import tabulate
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# Add parent directories to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "project_progress" / "part_3"))
sys.path.insert(0, str(ROOT / "project_progress" / "part_2"))

from indexing import load_processed_json, create_index

TEST_QUERIES = [
    "women cotton sweatshirt",
    "men blue jeans slim fit",
    "red dress party",
    "running shoes sports",
    "leather jacket black"
]


def train_sentence2vec_model(docs, vector_size=100, window=5, min_count=1, epochs=10):
    """
    Train Sentence2Vec using Word2Vec with n-gram context.
    Sentence2Vec learns sentence representations by training on word sequences
    with special handling for sentence boundaries and context.
    """
    sentences = []
    
    for doc in docs:
        tokens = []
        for field in ("title_tokens", "description_tokens", "product_details_tokens"):
            if field in doc:
                tokens.extend(doc[field])
        
        if tokens:
            # Add sentence boundary markers for better sentence-level learning
            sentences.append(['<s>'] + tokens + ['</s>'])
    
    # Train with CBOW (sg=0) which is better for sentence-level representations
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window,
                     min_count=min_count, workers=4, sg=0, epochs=epochs,
                     negative=5, ns_exponent=0.75)
    
    return model


def sentence2vec(tokens, model):
    """
    Convert sentence to vector using trained Sentence2Vec model.
    Uses weighted averaging with TF-IDF-like weighting and subsampling.
    """
    vectors = []
    weights = []
     
    # Add sentence boundaries
    tokens_with_boundary = ['<s>'] + tokens + ['</s>']
    
    for token in tokens_with_boundary:
        if token in model.wv:
            vectors.append(model.wv[token])
            # Weight by inverse frequency (like TF-IDF)
            freq = model.wv.get_vecattr(token, "count")
            weight = 1.0 / (1.0 + np.log(1.0 + freq))
            weights.append(weight)
    
    if not vectors:
        return np.zeros(model.vector_size)
    
    # Weighted average
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    return np.average(vectors, axis=0, weights=weights)


def rank_sentence2vec(query, model, docs, index):
    """Ranking using Sentence2Vec embeddings with weighted averaging."""
    from project_progress.part_1.preprocess import _clean_text, _tokenize_and_normalize

    cleaned = _clean_text(query)
    query_tokens = _tokenize_and_normalize(cleaned)

    # AND semantics using inverted index
    posting_lists = []
    for term in query_tokens:
        if term not in index:
            return []
        posting_lists.append(set(index[term]))

    matching_pids = set.intersection(*posting_lists)

    # Create sentence embedding for query
    query_vec = sentence2vec(query_tokens, model)

    # Create pid to document mapping
    pid_to_doc = {doc["pid"]: doc for doc in docs}

    results = []

    # Compute similarity only on AND-filtered docs
    for pid in matching_pids:
        doc = pid_to_doc.get(pid)
        if not doc:
            continue
        
        # Get document tokens
        tokens = []
        for field in ("title_tokens", "description_tokens", "product_details_tokens"):
            if field in doc:
                tokens.extend(doc[field])
        
        if not tokens:
            continue
        
        # Create sentence embedding for document
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

    # Train Sentence2Vec model
    print("Training Sentence2Vec model...")
    sentence2vec_model = train_sentence2vec_model(docs)
    print(f"  Vocabulary size: {len(sentence2vec_model.wv)} words")
    print(f"  Vector dimension: {sentence2vec_model.vector_size}\n")

    # Run queries
    all_results = {}
    for query in TEST_QUERIES:
        print("=" * 80)
        print(f"QUERY: '{query}'")
        print("=" * 80 + "\n")

        results = rank_sentence2vec(query, sentence2vec_model, docs, index)
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

if __name__ == "__main__":
    main()