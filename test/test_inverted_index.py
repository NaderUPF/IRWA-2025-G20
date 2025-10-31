import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "project_progress" / "part_2"))

from inverted_index import build_index_from_processed_data

def test_index():
    # Load processed data
    processed_file = ROOT / "data" / "processed_fashion.json"
    with open(processed_file, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
        
    # Build index (use a limit for quick runs if desired)
    index, _ = build_index_from_processed_data(processed_data, limit=500, show_progress=False)
    
    # Terms to inspect
    test_terms = ["shirt", "dress", "cotton"]
    
    # Print inverted index postings for each term (first 10)
    for term in test_terms:
        postings = index.get(term, [])
        print(f"\nFirst 10 Index results for term '{term}' ({len(postings)} total documents):")
        print(postings[:10])

if __name__ == "__main__":
    test_index()