import json
import sys
from pathlib import Path

# add project_progress/part_1 to sys.path so we can import preprocess.py
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "project_progress" / "part_1"))

from preprocess import preprocess_file

def test_preprocessing():
    data_dir = ROOT / "data"
    input_file = data_dir / "fashion_products_dataset.json"
    output_file = data_dir / "processed_fashion.json"
    
    # Run preprocessing
    preprocess_file(str(input_file), str(output_file))
    
    # Verify output
    with open(output_file, 'r', encoding='utf-8') as f:
        processed = json.load(f)
    
    # Print sample of first record
    if processed:
        print("\nFirst record preview:")
        record = processed[0]
        # show raw fields that are preserved; title/description tokens/clean are no longer produced
        print(f"Title: {record.get('title', '')}")
        print(f"Description: {record.get('description', '')}")
        print(f"Available keys in processed record: {(record.keys())}")
        print(f"\nTotal records processed: {len(processed)}")

if __name__ == "__main__":
    test_preprocessing()