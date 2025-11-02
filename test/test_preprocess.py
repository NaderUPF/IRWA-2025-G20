import json
import sys
from pathlib import Path

# add project_progress/part_1 to sys.path so we can import preprocess.py
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "project_progress" / "part_1"))

from preprocess import preprocess_file, SELECT_FIELDS

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
        
        # Show all fields that were preserved from SELECT_FIELDS
        print("Fields in processed record:")
        for field in SELECT_FIELDS:
            if field in record:
                value = record[field]
                # Truncate long values for display
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {field}: {value}")

if __name__ == "__main__":
    test_preprocessing()