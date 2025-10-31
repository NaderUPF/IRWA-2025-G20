from collections import defaultdict
from array import array
from typing import Dict, List, Tuple, Optional

def build_index_from_processed_data(
    processed_records: List[dict],
    limit: Optional[int] = None,
    show_progress: bool = False
) -> Tuple[Dict, Dict]:
    """
    Build inverted index from processed records (efficient build).
    - index (returned) has term -> list of [doc_id, array('I', positions)]
    - titles maps doc_id -> title
    - limit: process only the first `limit` records (useful for quick tests)
    - show_progress: print progress every 1000 documents
    """
    # temporary structure: term -> {doc_id: array(positions)}
    temp_index: Dict[str, Dict[str, array]] = defaultdict(dict)
    titles: Dict[str, str] = {}

    total = len(processed_records)
    if limit is not None:
        total = min(total, limit)

    for i, record in enumerate(processed_records[:total]):
        doc_id = str(record['pid'])
        titles[doc_id] = record.get('title', '')

        pos = 0
        for field in ('title_tokens', 'description_tokens'):
            tokens = record.get(field)
            if not isinstance(tokens, list):
                continue
            for token in tokens:
                postings_for_term = temp_index[token]
                if doc_id in postings_for_term:
                    postings_for_term[doc_id].append(pos)
                else:
                    postings_for_term[doc_id] = array('I', [pos])
                pos += 1

        if show_progress and (i + 1) % 1000 == 0:
            print(f"Indexed {i+1}/{total} docs")

    # convert to required final format: term -> list of [doc_id, array(...)] (preserve insertion order)
    index: Dict[str, List[List]] = {}
    for term, doc_map in temp_index.items():
        postings = [[doc_id, positions] for doc_id, positions in doc_map.items()]
        index[term] = postings

    return index, titles