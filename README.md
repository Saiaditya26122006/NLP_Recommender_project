# Novalnet — Book Recommender System (NLP)

A content-based book recommender using TF-IDF and Sentence-BERT embeddings.
Given a book title, author name, or free-text topic, the system returns the most
relevant book recommendations from a dataset of 6,810 books via a polished
Streamlit web interface.

---

## Project Motivation

Discovering new books that match your taste is hard — browsing by genre is too
broad, and search engines return bestsellers rather than truly similar content.
This project builds a recommender that understands the content of books (their
descriptions, categories, and authors) using NLP techniques, so users can get
meaningful suggestions from a simple text input.

The project is also a practical exercise in the full NLP pipeline: from raw text
cleaning to vectorization, similarity computation, and a deployable demo
interface — covering every design decision along the way.

---

## Dataset Description

**File:** `data/books.csv` — 6,810 books with the following columns:

| Column | Type | Description | Missing |
|---|---|---|---|
| isbn13 / isbn10 | str | Book identifiers | 0 |
| title | str | Book title | 0 |
| subtitle | str | Subtitle | 65% |
| authors | str | Author(s), semicolon-separated | 72 rows |
| categories | str | Genre/category label | 99 rows |
| thumbnail | str | Cover image URL | 329 rows |
| description | str | Full text description | 262 rows |
| published_year | int | Year of publication (1853–2019) | 6 rows |
| average_rating | float | Mean rating (0–5, avg ≈ 3.93) | 43 rows |
| num_pages | int | Page count (avg ≈ 348) | 43 rows |
| ratings_count | int | Number of ratings (median ≈ 1,018) | 43 rows |

**Fields used by the recommender:** `title`, `authors`, `categories`,
`description` — combined into a single weighted text field (`combined_text`).
Numerical fields (`average_rating`, `ratings_count`) are used for author-based
ranking and sidebar filtering.

**Fields not used in NLP:** `isbn13`, `isbn10`, `subtitle`, `thumbnail`,
`published_year`, `num_pages` — excluded because they carry no semantic content
relevant to content-based similarity. `thumbnail` is used in the UI only for
displaying book cover images.

---

## Dataset Analysis

Run the analysis script to generate charts and a printed summary:

```bash
python analysis/dataset_analysis.py
# → charts saved to analysis/charts/
```

**Key Findings:**

- **Ratings** — tightly clustered between 3.5 and 4.5 (mean 3.93, std 0.33).
  Almost no books below 3.0, reflecting a strong quality/popularity filter in
  the dataset.
- **Publication years** — 63% of books published in the 2000s decade. The
  corpus is largely contemporary, with a long tail back to 1853.
- **Categories** — 567 unique genres. Fiction dominates (2,588 books, 38%),
  followed by Juvenile Fiction (538) and Biography & Autobiography (401).
- **Authors** — 3,780 unique authors. Most prolific: Agatha Christie (37 books),
  Stephen King (36), William Shakespeare (35).
- **Missing values** — subtitle missing for 65% of rows (excluded from model).
  All other key fields are >98% complete; missing values are handled gracefully
  by `preprocess_data()`.

---

## Methodology

The system uses a content-based filtering approach with two NLP representations:

### 1. TF-IDF (title-based similarity)

Each book's title, authors, categories, and description are combined into a
single `combined_text` field with weights: **title × 3, categories × 2,
authors × 1, description × 1** — so genre and title dominate.

Text is cleaned: lowercased, punctuation/numbers removed, NLTK English
stopwords removed.

`TfidfVectorizer` (scikit-learn) maps each book to a 10,000-dimensional sparse
vector. TF-IDF gives higher weight to terms that are distinctive for a book but
rare across the corpus.

Pairwise cosine similarity is precomputed once at startup into an N×N matrix.
At request time, looking up the most similar books is O(1) index + sort.

### 2. Sentence-BERT (semantic / query-based)

The `all-MiniLM-L6-v2` model (Sentence Transformers) encodes each book's
`combined_text` into a dense 384-dimensional embedding that captures semantic
meaning — synonyms and paraphrases map to nearby vectors.

The same model encodes the user's free-text query, and cosine similarity is
computed against all precomputed book embeddings.

Used when the input is a topic or phrase (e.g. "fantasy magic", "books about
travel") rather than an exact title.

### Smart Routing

`smart_recommend(user_input)` routes automatically:

- **Exact title match** → TF-IDF similarity (most similar books by content)
- **Author name match** → author search, sorted by `average_rating`
- **Everything else** → Sentence-BERT semantic query

### Why cosine similarity?

TF-IDF vectors are high-dimensional and sparse; cosine similarity is invariant
to document length (a short description and a long one are comparable), making
it the standard choice for text similarity.

---

## System Architecture

The system is designed in clear layers: each layer has a single responsibility
and passes structured output to the next. All heavy computation (load,
preprocessing, vectorization, similarity) happens once at startup; the
recommendation path is a lightweight lookup and rank.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  GUI Layer (Streamlit)                                                    │
│  Search bar + "Search" button → sidebar filters (category, rating, year) │
│  → 3-column card grid (cover image, title, author, score, description)   │
│  → search mode badge (Title / Author / Semantic)                         │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Recommendation Layer (BookRecommender.smart_recommend)                  │
│  Route input → title / author / query → rank → return top_n             │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Similarity Engine                                                        │
│  TF-IDF: precomputed N×N cosine matrix  |  SBERT: precomputed embeddings │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Vectorization Layer (TF-IDF + Sentence-BERT)                            │
│  TfidfVectorizer: combined_text → sparse matrix (N × 10,000)            │
│  SentenceTransformer: combined_text → dense matrix (N × 384)            │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Preprocessing Layer                                                      │
│  load_data → weighted combined_text → clean_text (normalize, stopwords)  │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              data/books.csv
```

### Layer Details

**1. Preprocessing Layer** (`src/preprocessing.py`)
- `load_data(path)` reads `data/books.csv`, validates required columns, fills
  missing values with empty strings.
- `preprocess_data(df)` builds `combined_text`: title × 3 + categories × 2 +
  authors × 1 + description × 1.
- `clean_text()` lowercases, strips punctuation and numbers, removes NLTK
  English stopwords.

**2. Vectorization Layer** (`src/recommender.py` — `BookRecommender.__init__`)
- TF-IDF: `TfidfVectorizer(max_features=10_000, ngram_range=(1,2), max_df=0.95)`
  fitted once at startup.
- Sentence-BERT: `all-MiniLM-L6-v2` encodes all books into 384-dim embeddings,
  stored in `book_embeddings`.

**3. Similarity Engine** (`src/recommender.py` — `BookRecommender.__init__`)
- TF-IDF path: `cosine_similarity(tfidf_matrix, tfidf_matrix)` → dense N×N
  matrix precomputed at startup.
- SBERT path: `cosine_similarity(query_embedding, book_embeddings)` at request
  time (query cannot be precomputed).

**4. Recommendation Layer** (`src/recommender.py` — `BookRecommender`)
- Title path: case-insensitive match → row index → similarity row → exclude
  self → sort → top_n.
- Author path: substring match on `authors` → sort by `average_rating` → top_n.
- Query path: encode with SBERT → cosine similarity → top_n.

**5. GUI Layer** (`src/app.py`)
- Stack: **Streamlit**. Dark navy/gold theme, wide layout.
- Sidebar: result count slider (3–20), category multiselect, min rating slider,
  year range slider.
- Search: text input + Search button → `smart_recommend()` → `enrich_results()`
  → sidebar filters applied → 3-column card grid.
- Each card: book cover image (Google Books thumbnail, emoji fallback), category
  badge, title, author, year, rating, similarity score bar, description snippet.
- Search mode badge: green (Title match), blue (Author search), gold (Semantic).

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit app (from project root)
streamlit run src/app.py
# → opens http://localhost:8501 in your browser

# 3. Run tests
python test_preprocessing.py
python test_recommender.py
python test_semantic.py

# 4. Run dataset analysis
python analysis/dataset_analysis.py
```

---

## Example Usage

**By book title** — returns books with similar content:

```
Input:  Gilead
Output: Crossing to Safety  | Wallace Stegner      | score=0.4821
        Housekeeping         | Marilynne Robinson   | score=0.4103
        ...
```

**By author name** — returns that author's books sorted by rating:

```
Input:  Agatha Christie
Output: And Then There Were None       | Agatha Christie | rating=4.27
        Murder on the Orient Express   | Agatha Christie | rating=4.17
        ...
```

**By free-text topic** — semantic search via Sentence-BERT:

```
Input:  fantasy magic
Output: Harry Potter and the Philosopher's Stone | J.K. Rowling     | score=0.6412
        The Name of the Wind                     | Patrick Rothfuss | score=0.6204
        ...
```

---

## Limitations

- **Content-only:** Uses only book metadata. No user history or preferences —
  no personalization.
- **Exact title match:** TF-IDF recommendations require a case-insensitive exact
  title match. No fuzzy search.
- **TF-IDF bag-of-words:** No notion of word order or semantics beyond n-grams.
  Synonyms not captured — addressed by the SBERT query path.
- **Single language:** Preprocessing and stopwords are English-only.
- **In-memory:** Full dataset and N×N similarity matrix live in one process.
  Scaling to millions of books would require approximate nearest-neighbor search
  (e.g. FAISS).
- **Cold start for new books:** New books require reloading data and recomputing
  TF-IDF and similarity — no incremental update.
- **No diversity or explanation:** Ranking is by similarity only. No genre
  spread or explainability.

---

## Future Improvements

- **Fuzzy / partial title search:** Integrate NearestNeighbors on title
  embeddings so users can type partial titles.
- **Hybrid with collaborative filtering:** Combine content-based scores with
  matrix factorization for personalized rankings.
- **Caching and incremental updates:** Cache similarity matrix on disk; support
  incremental indexing for new books.
- **API layer:** Expose the recommender as a FastAPI REST service; keep
  Streamlit as one consumer.
- **Diversity and re-ranking:** Apply MMR or category spread after top-k
  retrieval.
- **Configurable weights:** Move field weights and TF-IDF hyperparameters into
  a config file (YAML/env) for experimentation.
- **Cloud deployment:** Host on Streamlit Cloud or AWS for public access without
  local setup.

---

## Team

| Role | Responsibility |
|---|---|
| Data & Preprocessing Lead | Dataset analysis, cleaning pipeline, combined_text logic |
| NLP & Recommender Lead | TF-IDF engine, Sentence-BERT, cosine similarity, smart routing |
| GUI & Integration Lead | Streamlit interface, end-to-end wiring, demo readiness |
| Evaluation & Presentation Lead | Testing, README/docs, slide deck, Q&A preparation |

---

## Technical Summary

```
CSV (data/books.csv)
    → Preprocessing: load_data() → preprocess_data() [weighted combined_text + clean_text]
    → Vectorization: TfidfVectorizer.fit_transform(combined_text) + SBERT.encode(combined_text)
    → Similarity: cosine_similarity(tfidf_matrix) → N×N matrix  |  book_embeddings stored
    → Stored in BookRecommender (once at startup)

Request: (user_input, top_n)
    → smart_recommend: route to title / author / query path
    → Recommendation: lookup / encode → rank → top_n with metadata
    → GUI: enriched results → Streamlit card grid with sidebar filters
```
