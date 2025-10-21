import argparse
import json
import os
import re
import unicodedata
from typing import Dict, List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Setup stopwords
STOPWORDS = set(stopwords.words("english"))

FASHION_STOPWORDS = {
    "size", "sizes", "colour", "color", "fit", "model", "made", "wear", "new", "sale", "style", "sku"
}
STOPWORDS |= FASHION_STOPWORDS

Stemmer = PorterStemmer() if PorterStemmer is not None else None

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


def _normalize_unicode(text: str) -> str:
    if not text:
        return ""
    if unidecode:
        return unidecode(text)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _clean_text(text: str) -> str:
    text = text.lower()
    text = _RE_HTML.sub(" ", text)
    text = _normalize_unicode(text)
    text = _RE_NON_ALPHA.sub(" ", text)
    return _RE_MULTI_WS.sub(" ", text).strip()


def _tokenize_and_normalize(cleaned: str) -> List[str]:
    tokens = [t for t in cleaned.split() if len(t) > 1 and t not in STOPWORDS]
    if Stemmer is not None:
        processed = []
        for t in tokens:
            try:
                t = Stemmer.stem(t)
            except Exception:
                pass
            if len(t) > 1:
                processed.append(t)
        return processed
    # fallback: return tokens as-is
    return tokens


def clean_and_tokenize(text: str) -> Tuple[List[str], str]:
    """
    Return (tokens_list, joined_clean_text).
    Steps: lowercase, strip HTML, normalize accents, remove non-alpha, collapse whitespace,
    tokenize, remove stopwords, lemmatize+stem when available.
    """
    if not text:
        return [], ""
    cleaned = _clean_text(text)
    tokens = _tokenize_and_normalize(cleaned)
    return tokens, " ".join(tokens)


def preprocess_record(rec: Dict, text_fields: Tuple[str, ...] = ("title", "description")) -> Dict:
    """
    Build processed record containing:
    - the requested SELECT_FIELDS (only if present in source record)
    - for each text_field: <field>_tokens and <field>_clean
    Other source fields are not included to keep output focused for future queries.
    """
    out: Dict = {}
    # include only the selected fields if they exist in the source record
    for field in SELECT_FIELDS:
        if field in rec:
            out[field] = rec[field]

    # add cleaned/tokenized versions for specified text fields (title/description by default)
    for field in text_fields:
        raw = rec.get(field) or ""
        tokens, joined = clean_and_tokenize(raw)
        out[f"{field}_tokens"] = tokens
        out[f"{field}_clean"] = joined

    return out


def preprocess_file(input_path: str, output_path: str, text_fields: Tuple[str, ...] = ("title", "description")) -> None:
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
    parser.add_argument("-f", "--fields", nargs="+", default=["title", "description"],
                        help="Text fields to preprocess (default: title description)")
    args = parser.parse_args()
    preprocess_file(args.input, args.output, text_fields=tuple(args.fields))


if __name__ == "__main__":
    main()