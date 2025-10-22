import argparse
import json
import os
import re

import nltk
from nltk.stem import PorterStemmer

# ensure stopwords are available
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

Stemmer = PorterStemmer()

# Regex compiled once
_RE_HTML = re.compile(r"<[^>]+>")
_RE_NON_ALPHA = re.compile(r"[^a-zA-Z\s]")
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


def _clean_text(text: str) -> str:
    text = text.lower()
    text = _RE_HTML.sub(" ", text)
    text = _RE_NON_ALPHA.sub(" ", text)
    return _RE_MULTI_WS.sub(" ", text).strip()


def _tokenize_and_normalize(cleaned: str) -> list:
    tokens = [t for t in cleaned.split() if len(t) > 1 and t not in STOPWORDS]
    if Stemmer is not None:
        return [Stemmer.stem(t) for t in tokens]
    return tokens


def clean_and_tokenize(text: str) -> tuple:
    if not text:
        return [], ""
    cleaned = _clean_text(text)
    tokens = _tokenize_and_normalize(cleaned)
    return tokens, " ".join(tokens)


def preprocess_record(rec: dict, text_fields: tuple = ("title", "description")) -> dict:
    """
    Build processed record containing only the requested SELECT_FIELDS (if present).
    """
    out: dict = {}
    # include only the selected fields if they exist in the source record
    for field in SELECT_FIELDS:
        if field in rec:
            out[field] = rec[field]

    return out


def preprocess_file(input_path: str, output_path: str, text_fields: tuple = ("title", "description")) -> None:
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

    processed = [preprocess_record(r, text_fields=text_fields) for r in records]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(processed)} records -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess fashion product JSON (title, description).")
    parser.add_argument("-i", "--input", required=True, help="Path to input JSON (fashion_products_dataset.json)")
    parser.add_argument("-o", "--output", default="processed_fashion.json", help="Output JSON path")
    args = parser.parse_args()
    preprocess_file(args.input, args.output)


if __name__ == "__main__":
    main()