# Information Retrieval and Web Analytics (IRWA) - Final Project template

<table>
  <tr>
    <td style="vertical-align: top;">
      <img src="static/image.png" alt="Project Logo"/>
    </td>
    <td style="vertical-align: top;">
      This repository contains the template code for the IRWA Final Project - Search Engine with Web Analytics.
      The project is implemented using Python and the Flask web framework. It includes a simple web application that allows users to search through a collection of documents and view analytics about their searches.
    </td>
  </tr>
</table>

---

## Project Structure

```
/irwa-search-engine
├── myapp                # Contains the main application logic
├── templates            # Contains HTML templates for the Flask application
├── static               # Contains static assets (images, CSS, JavaScript)
├── data                 # Contains the dataset file (fashion_products_dataset.json)
├── project_progress     # Contains your solutions for Parts 1, 2, and 3 of the project
├── .env                 # Environment variables for configuration (e.g., API keys)
├── .gitignore           # Specifies files and directories to be ignored by Git
├── LICENSE              # License information for the project
├── requirements.txt     # Lists Python package dependencies
├── web_app.py           # Main Flask application
└── README.md            # Project documentation and usage instructions
```

---

## To download this repo locally

Open a terminal console and execute:

```
cd <your preferred projects root directory>
git clone https://github.com/trokhymovych/irwa-search-engine.git
```

## Setting up the Python environment (only for the first time you run the project)

### Install virtualenv

Setting up a virtualenv is recommended to isolate the project dependencies from other Python projects on your machine.
It allows you to manage packages on a per-project basis, avoiding potential conflicts between different projects.

In the project root directory execute:

```bash
pip3 install virtualenv
virtualenv --version
```

### Prepare virtualenv for the project

In the root of the project folder run to create a virtualenv named `irwa_venv`:

```bash
virtualenv irwa_venv
```

If you list the contents of the project root directory, you will see that it has created a new folder named `irwa_venv` that contains the virtualenv:

```bash
ls -l
```

The next step is to activate your new virtualenv for the project:

```bash
source irwa_venv/bin/activate
```

or for Windows...

```cmd
irwa_venv\Scripts\activate.bat
```

This will load the python virtualenv for the project.

### Installing Flask and other packages in your virtualenv

Make sure you are in the root of the project folder and that your virtualenv is activated (you should see `(irwa_venv)` in your terminal prompt).
And then install all the packages listed in `requirements.txt` with:

```bash
pip install -r requirements.txt
```

If you need to add more packages in the future, you can install them with pip and then update `requirements.txt` with:

```bash
pip freeze > requirements.txt
```

Enjoy!

## Usage:

0. Put the data file `fashion_products_dataset.json` in the `data` folder. It will be provided to you by the instructor.
1. As for Parts 1, 2, and 3 of the project, please use the `project_progress` folder to store your solutions. Each part should contain `.pdf` file with your report and `.ipynb` (Jupyter Notebook) file with your code for solution and `README.md` with explanation of the content and instructions for results reproduction.
2. For the Part 4, of the project, you should build a web application using Flask that allows users to search through a collection of documents and view analytics about their searches. You should work mailnly in the `web_app.py` file `myapp` and `templates` folders. Feel free to change any code or add new files as needed. The provided code is just a starting point to help you get started quickly.
3. Make sure to update the `.env` file with your Groq API key (can be found [here](https://groq.com/), the free version is more than enough for our purposes) and any other necessary configurations. IMPORTANT: Do not share your `.env` file publicly as it contains sensitive information. It is included in `.gitignore` to prevent accidental commits. (It should never be included in the repo and appears here only for demonstration purposes).

## Environment variables (.env)

Create a file named `.env` in the project root with the following keys. These are read by `web_app.py`, the search engine loader and the RAG generator:

- `SECRET_KEY`: Flask secret key used to sign session cookies. Example: `SECRET_KEY="a-very-secret-value"`
- `DEBUG`: Optional (True/False). Example: `DEBUG=True`
- `SESSION_COOKIE_NAME`: Optional name for the Flask session cookie. Example: `SESSION_COOKIE_NAME="IRWA_SEARCH_ENGINE"`
- `DATA_FILE_PATH` or `PROCESSED_DATA_PATH`: Path to the processed corpus JSON. Example: `DATA_FILE_PATH="data/processed_fashion.json"`
- `GROQ_API_KEY`: (required for RAG) Your Groq API key. Example: `GROQ*API_KEY="gsk*..."
- `GROQ_MODEL`: (optional) Model name to use with Groq. Example: `GROQ_MODEL="llama-3.1-8b-instant"`

Example `.env` (do not commit):

```dotenv
SECRET_KEY="replace_me_with_a_secure_random_value"
DEBUG=True
SESSION_COOKIE_NAME="IRWA_SEARCH_ENGINE"
DATA_FILE_PATH="data/processed_fashion.json"

GROQ_API_KEY="gsk_your_api_key_here"
GROQ_MODEL="llama-3.1-8b-instant"
```

Security notes:

- Never commit `.env` to source control. If you accidentally committed secrets, rotate them immediately and remove the file from the repository history (see the project issues or use `git filter-repo` / BFG). You can stop tracking the file locally with:

```bash
git rm --cached .env
echo ".env" >> .gitignore
git add .gitignore
git commit -m "Remove .env from repo and add to .gitignore"
```

Restart the Flask app after editing `.env` so changes are picked up.

### Part 1: Data Preprocessing & Exploratory Data Analysis

This part implements text preprocessing (cleaning, tokenization, stemming) and exploratory data analysis on the fashion product dataset.

#### Running Preprocessing

To preprocess the raw dataset:

```bash
cd project_progress/part_1
python preprocess.py -i ../../data/fashion_products_dataset.json -o ../../data/processed_fashion.json
```

This will:

- Load the raw fashion products dataset
- Clean text (lowercase, remove HTML tags, remove punctuation, normalize whitespace)
- Tokenize and stem text fields (title, description, product_details)
- Remove stopwords from tokenized fields
- Convert numeric fields (prices, ratings) to proper numeric types
- Save processed data to `data/processed_fashion.json`

#### Testing Preprocessing

To validate the preprocessing results:

```bash
cd test
python test_preprocess.py
```

This will:

- Load the processed dataset
- Verify all required fields are present
- Check data types and formats
- Display sample processed records

#### Exploratory Data Analysis

To view and run the data analysis:

```bash
cd project_progress/part_1
jupyter notebook analysis.ipynb
```

This notebook includes:

- Dataset statistics and distribution analysis
- Text field analysis (word frequencies, common terms)
- Category and brand distribution
- Price and rating analysis
- Visualization of key insights

**Key Components:**

- `project_progress/part_1/preprocess.py` - Text preprocessing implementation
- `project_progress/part_1/analysis.ipynb` - Jupyter notebook with EDA
- `test/test_preprocess.py` - Preprocessing validation script

### Part 2: Indexing, Search, and Evaluation

This part implements an inverted index, TF-IDF ranking, and comprehensive evaluation metrics for the search system.

#### Building the Index

To build the inverted index from the preprocessed data:

```bash
cd test
python test_indexing.py
```

This will:

- Load the preprocessed documents from `data/processed_fashion.json`
- Build the inverted index with stemmed terms
- Compute TF-IDF weights for all documents
- Save the index to `data/inverted_index.json`
- Test search functionality with sample queries

#### Running Evaluation

To evaluate the search system with validation queries:

```bash
cd test
python test_evaluation.py
```

This will:

- Load ground truth labels from `data/validation_labels.csv`
- Run the validation queries through the search system
- Compute all evaluation metrics (P@K, R@K, F1@K, AP@K, MAP, MRR, NDCG@K)
- Display detailed results including top-ranked documents and metric scores

**Evaluation Metrics Computed:**

- **Precision@K (P@K)**: Proportion of relevant documents in top K results
- **Recall@K (R@K)**: Proportion of all relevant documents found in top K
- **F1-Score@K**: Harmonic mean of precision and recall
- **Average Precision@K (AP@K)**: Precision averaged at positions of relevant documents
- **Mean Average Precision (MAP)**: Overall system quality across all queries
- **Mean Reciprocal Rank (MRR)**: Average position of first relevant result
- **NDCG@K**: Normalized Discounted Cumulative Gain for ranking quality

**Output Format:**

```
Query 1: 'women full sleeve sweatshirt cotton'
   Top 10 Results:
   1. [✓] Full Sleeve Printed Women Sweatshirt
      Score: 3.4699
   ...

   @ K=5:
      Precision@5  : 0.000
      Recall@5     : 0.000
      F1-Score@5   : 0.000
      AP@5         : 0.000
      RR@5         : 0.000
      NDCG@5       : 0.000
```

**Key Components:**

- `project_progress/part_2/indexing.py` - Inverted index and TF-IDF ranking implementation
- `project_progress/part_2/evaluation.py` - All evaluation metrics implementation
- `test/test_indexing.py` - Index building and search testing
- `test/test_evaluation.py` - Comprehensive evaluation script

### Part 3: Ranking Algorithms

This part implements and compares multiple ranking algorithms including TF-IDF, BM25, Custom Score, Word2Vec, and Sentence2Vec for semantic search.

#### Running Algorithm Comparisons

To compare all 4 ranking algorithms side-by-side:

```bash
cd project_progress/part_3
python compare_rankings.py
```

This will:

- Load the preprocessed documents
- Build all necessary indices (inverted index, TF-IDF vectors, BM25 index, Word2Vec model, metadata)
- Run 5 test queries through all 4 algorithms (TF-IDF, BM25, Custom Score, Word2Vec)
- Display top-20 results for each algorithm in a comparison table
- Show ranking overlap analysis
- Display summary of algorithm characteristics

#### Running Word2Vec Ranking Only

To run only the Word2Vec + Cosine Similarity ranking:

```bash
cd project_progress/part_3
python word2vec_ranking.py
```

This will:

- Train Word2Vec model on the product corpus (100-dimensional embeddings)
- Represent text as averaged word vectors: `(v1 + v2 + ... + vn) ÷ n`
- Rank documents using cosine similarity between query and document vectors
- Return top-20 results for each of the 5 queries
- Display results in a formatted table

#### Running Sentence2Vec Ranking Only

To run only the Sentence2Vec + Cosine Similarity ranking:

```bash
cd project_progress/part_3
python sentence2vec_ranking.py
```

This will:

- Train Sentence2Vec model with sentence boundary markers (`<s>`, `</s>`)
- Use weighted averaging with inverse frequency weighting (TF-IDF-like)
- Represent sentences with frequency-weighted word vectors
- Rank documents using cosine similarity between query and document embeddings
- Return top-20 results for each of the 5 queries
- Display results in a formatted table

**Implemented Algorithms:**

1. **TF-IDF + Cosine Similarity** (Corrected from Part 2)

   - Computes TF-IDF weights for documents and queries
   - Ranks by cosine similarity between query and document vectors
   - Enforces AND semantics (all query terms must appear in document)

2. **BM25** (using rank-bm25 library)

   - Probabilistic ranking with term frequency saturation
   - Document length normalization
   - Industry-standard algorithm (default params: k1=1.5, b=0.75)

3. **Custom Score** (Text + Quality + Value)

   - Combines: 60% text relevance + 25% quality + 15% value
   - Quality score: product rating × stock availability
   - Value score: discount percentage
   - Domain-specific for e-commerce

4. **Word2Vec + Cosine Similarity**

   - Trains Word2Vec embeddings on corpus (100 dimensions)
   - Represents text as simple averaged word vectors
   - Ranks by semantic similarity using cosine distance
   - Captures synonyms and related terms

5. **Sentence2Vec + Cosine Similarity**
   - Trains with sentence boundary markers for better context
   - Uses weighted averaging with inverse frequency weighting
   - Weights rare words higher than common words (TF-IDF-like)
   - Better sentence-level semantic representation than simple averaging

**Test Queries:**

- "women cotton sweatshirt"
- "men blue jeans slim fit"
- "red dress party"
- "running shoes sports"
- "leather jacket black"

**Output Format:**

```
QUERY: 'women cotton sweatshirt'
========================================================

Top 20 Results:

+------+------------------+--------+------------------+--------+------------------+--------+------------------+--------+
| Rank | TF-IDF+Cosine    | Score  | BM25             | Score  | Custom           | Score  | Word2Vec         | Score  |
+======+==================+========+==================+========+==================+========+==================+========+
| 1    | Product Title... | 0.8543 | Product Title... | 12.456 | Product Title... | 0.7234 | Product Title... | 0.9123 |
| ...  | ...              | ...    | ...              | ...    | ...              | ...    | ...              | ...    |
+------+------------------+--------+------------------+--------+------------------+--------+------------------+--------+

Ranking Overlap Analysis (Top 5):
  TF-IDF ∩ BM25:      4/5 documents
  TF-IDF ∩ Custom:    3/5 documents
  TF-IDF ∩ Word2Vec:  2/5 documents
  BM25 ∩ Custom:      3/5 documents
  BM25 ∩ Word2Vec:    2/5 documents
  Custom ∩ Word2Vec:  1/5 documents
  All Four:           1/5 documents
```

**Key Components:**

- `project_progress/part_3/ranking_algorithms.py` - Core implementations of TF-IDF, BM25, Custom Score, and Word2Vec algorithms
- `project_progress/part_3/compare_rankings.py` - Side-by-side comparison tool for all 4 algorithms
- `project_progress/part_3/word2vec_ranking.py` - Standalone Word2Vec ranking script (simple averaging)
- `project_progress/part_3/sentence2vec_ranking.py` - Standalone Sentence2Vec ranking script (weighted averaging)

## Search Engine Features

To use the search engine with different ranking algorithms:

1. Start the web application: `python web_app.py`
2. Open your browser to: `http://127.0.0.1:8088/`
3. Available ranking algorithms can be selected from the dropdown menu in the search interface
