import argparse
import json
import os
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# ensure stopwords are available
nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Regex compiled once
_RE_HTML = re.compile(r"<[^>]+>")
_RE_MULTI_WS = re.compile(r"\s+")

# Fields required in final output
SELECT_FIELDS = (
    "pid",
    "title",
    "description",
    "brand",
    "category",
    "sub_category",
    "product_details",
    "seller",
    "out_of_stock",
    "selling_price",
    "discount",
    "actual_price",
    "average_rating",
    "url",
)


def _clean_text(text: str) -> str:
    """Clean text by removing HTML and normalizing whitespace"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = _RE_HTML.sub(" ", text)
    text = _RE_MULTI_WS.sub(" ", text).strip()
    return text


def _tokenize_and_normalize(text: str) -> list:
    """Split text into tokens and apply stemming"""
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1 and t not in STOPWORDS]
    return [stemmer.stem(t) for t in tokens]


def clean_and_tokenize(text: str) -> tuple:
    """Clean and tokenize text, return (tokens, cleaned_text)"""
    if not text:
        return [], ""
    cleaned = _clean_text(text)
    tokens = _tokenize_and_normalize(cleaned)
    return tokens, " ".join(tokens)


def preprocess_record(rec: dict) -> dict:
    """Process a single record, preserving required fields"""
    out = {}
    
    # Copy SELECT_FIELDS if present
    for field in SELECT_FIELDS:
        if field in rec:
            # Convert numeric fields to float
            if field in ('selling_price', 'discount', 'actual_price', 'average_rating'):
                try:
                    out[field] = float(rec[field])
                except (ValueError, TypeError):
                    out[field] = 0.0
            # Convert out_of_stock to boolean
            elif field == 'out_of_stock':
                out[field] = bool(rec[field])
            # Process text fields
            elif field in ('title', 'description', 'product_details'):
                tokens, cleaned = clean_and_tokenize(rec[field])
                out[f"{field}_tokens"] = tokens
                out[field] = rec[field]  # preserve original
            # Clean but don't tokenize these fields
            elif field in ('brand', 'category', 'sub_category', 'seller'):
                out[field] = _clean_text(rec[field])
            # Copy other fields as-is
            else:
                out[field] = rec[field]
    
    return out


def preprocess_file(input_path: str, output_path: str) -> None:
    """Load JSON, preprocess records, and save results"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "items" in data:
        records = data["items"]
    elif isinstance(data, list):
        records = data
    else:
        records = list(data.values())

    processed = [preprocess_record(r) for r in records]
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(processed)} records -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess fashion product JSON.")
    parser.add_argument("-i", "--input", required=True, help="Path to input JSON (fashion_products_dataset.json)")
    parser.add_argument("-o", "--output", default="processed_fashion.json", help="Output JSON path")
    args = parser.parse_args()
    preprocess_file(args.input, args.output)


if __name__ == "__main__":
    main()