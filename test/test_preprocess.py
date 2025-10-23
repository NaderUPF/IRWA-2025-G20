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
        
        print(f"\nTotal records processed: {len(processed)}")
        
        # Additional validation
        print(f"\nValidation checks:")
        print(f"- All records are dictionaries: {all(isinstance(r, dict) for r in processed)}")
        print(f"- Average fields per record: {sum(len(r) for r in processed) / len(processed):.1f}")
        
        # Check that only SELECT_FIELDS are present
        all_keys = set()
        for record in processed:
            all_keys.update(record.keys())
        unexpected_keys = all_keys - set(SELECT_FIELDS)
        print(f"- Only SELECT_FIELDS present: {len(unexpected_keys) == 0}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")

if __name__ == "__main__":
    test_preprocessing()