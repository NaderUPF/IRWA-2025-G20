import argparse
import json
import os
import re
import string

import nltk
from nltk.stem import PorterStemmer

# ensure stopwords are available
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Regex compiled once
_RE_HTML = re.compile(r"<[^>]+>")
_RE_MULTI_WS = re.compile(r"\s+")

# Fields required in the final output (when present in source records)
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

# Define which fields should be cleaned and tokenized
TEXT_FIELDS_TO_CLEAN = ("title", "description", "product_details")

# Define which fields should only be cleaned but not tokenized
TEXT_FIELDS_CLEAN_ONLY = ("brand", "category", "sub_category", "seller")

# Define which fields should be converted to numeric
NUMERIC_FIELDS = ('selling_price', 'discount', 'actual_price', 'average_rating')


def _clean_text(text: str) -> str:
    text = text.lower()  # Transform in lowercase
    text = _RE_HTML.sub(" ", text)  # Remove HTML tags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = _RE_MULTI_WS.sub(" ", text).strip()  # Normalize whitespace
    return text


def _tokenize_and_normalize(cleaned: str) -> list:
    text = cleaned.split()  # Tokenize the text to get a list of terms
    text = [x for x in text if x not in STOPWORDS]  # eliminate the stopwords
    text = [stemmer.stem(x) for x in text]  # perform stemming
    return text


def clean_and_tokenize(text: str) -> tuple:
    if not text:
        return [], ""
    cleaned = _clean_text(text)
    tokens = _tokenize_and_normalize(cleaned)
    return tokens, " ".join(tokens)


def clean_only(text: str) -> str:
    """
    Only clean the text without tokenizing - for fields like brand, category, etc.
    """
    if not text:
        return ""
    return _clean_text(text)


def _convert_to_numeric(value):
    """
    Converts a value to float, handling various string formats and errors.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    
    # Handle string values
    if isinstance(value, str):
        # Remove currency symbols, commas, etc.
        cleaned_value = re.sub(r'[^\d.-]', '', value.strip())
        if cleaned_value:
            try:
                return float(cleaned_value)
            except (ValueError, TypeError):
                return None
    return None


def preprocess_record(rec: dict) -> dict:
    """
    Build processed record containing only the requested SELECT_FIELDS (if present).
    Clean and tokenize text fields, convert numeric fields to float.
    """
    out: dict = {}
    
    # include only the selected fields if they exist in the source record
    for field in SELECT_FIELDS:
        if field in rec:
            # For text fields that need full cleaning and tokenization
            if field in TEXT_FIELDS_TO_CLEAN and rec[field]:
                tokens, cleaned_text = clean_and_tokenize(str(rec[field]))
                # Store both the cleaned text and tokens
                out[field] = cleaned_text
                out[f"{field}_tokens"] = tokens
            
            # For text fields that only need cleaning (no tokenization)
            elif field in TEXT_FIELDS_CLEAN_ONLY and rec[field]:
                cleaned_text = clean_only(str(rec[field]))
                out[field] = cleaned_text
                # No tokens stored for these fields
            
            # For numeric fields, convert to float
            elif field in NUMERIC_FIELDS and rec[field] is not None:
                numeric_value = _convert_to_numeric(rec[field])
                if numeric_value is not None:
                    out[field] = numeric_value
                else:
                    # Keep original value if conversion fails
                    out[field] = rec[field]
            
            # For all other fields, copy as-is
            else:
                out[field] = rec[field]
    
    return out


def preprocess_file(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        records = data["items"]
    elif isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = list(data.values())
    else:
        records = []

    # Only process records that have at least one of the SELECT_FIELDS
    filtered_records = []
    for record in records:
        if any(field in record for field in SELECT_FIELDS):
            filtered_records.append(record)

    processed = [preprocess_record(r) for r in filtered_records]

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