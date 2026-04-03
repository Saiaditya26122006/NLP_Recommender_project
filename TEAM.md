# Team Organization

## Roles & Responsibilities

| # | Role | Owner | Core Responsibility |
|---|------|-------|---------------------|
| 1 | **Data & Preprocessing Lead** | Sai Aditya Talluri | Dataset analysis, cleaning pipeline, combined_text logic |
| 2 | **NLP & Recommender Lead** | Martina | TF-IDF engine, Sentence-BERT, cosine similarity, smart routing |
| 3 | **GUI & Integration Lead** | Richard | Streamlit interface, end-to-end wiring, demo readiness |
| 4 | **Evaluation & Presentation Lead** | Eliott | Testing, README/docs, slide deck, Q&A preparation |

> Everyone must understand the full pipeline — the professor may ask any member about any part.

---

## Task Breakdown

### Sai Aditya Talluri — Data & Preprocessing
**Owns:** `src/preprocessing.py`, `data/books.csv`

- [x] Load and explore dataset (6,810 books, 12 columns)
- [x] Implement `load_data()` with column validation
- [x] Design weighted `combined_text` (title×3, categories×2, authors×1, description×1)
- [x] Implement `clean_text()` — lowercase, punctuation removal, stopwords, whitespace
- [x] Implement `preprocess_data()` applying weights + cleaning
- [x] Prepare **dataset analysis script** — distributions of ratings, categories, page counts, missing values (`analysis/dataset_analysis.py`)
- [ ] Be able to explain: *"Why did you weight title×3 and not description×3?"*

---

### Martina — NLP & Recommender Engine
**Owns:** `src/recommender.py`

- [x] Implement `BookRecommender.__init__()` — TF-IDF matrix + cosine similarity matrix
- [x] Implement `recommend()` — title-based TF-IDF lookup
- [x] Integrate Sentence-BERT (`all-MiniLM-L6-v2`) — precompute book embeddings
- [x] Implement `recommend_by_query()` — semantic search via SBERT
- [x] Implement `recommend_by_author()` — author substring match + rating sort
- [x] Implement `smart_recommend()` — routing logic (title / author / query)
- [ ] Be able to explain: *"What is TF-IDF? What is cosine similarity? Why not Euclidean distance?"*
- [ ] Be able to explain: *"How does Sentence-BERT differ from TF-IDF semantically?"*

---

### Richard — GUI & Integration
**Owns:** `src/app.py`

- [x] Build Streamlit interface with dark navy/gold theme, wide layout
- [x] Implement sidebar filters: result count, category multiselect, min rating, year range
- [x] Implement search bar + "Search" button wired to `smart_recommend()`
- [x] Implement 3-column book card grid with cover images, badges, similarity bars
- [x] Implement search mode badge (Title / Author / Semantic)
- [x] Implement `enrich_results()` — joins results back to full DataFrame for metadata
- [x] Handle edge cases in UI (no results, missing thumbnails)
- [ ] Polish the demo — test all 3 input modes (title / author / topic)
- [ ] Prepare **live demo script** for the 15-minute presentation
- [ ] Ensure the app launches cleanly (`streamlit run src/app.py`) for demo day
- [ ] Be able to explain: *"Why Streamlit over Gradio? What does the GUI layer do vs. the recommender layer?"*

---

### Eliott — Evaluation, Docs & Presentation
**Owns:** `test_*.py`, `README.md`, `TEAM.md`, slides

- [x] Write `test_preprocessing.py` — unit tests for `clean_text`, `preprocess_data`
- [x] Write `test_recommender.py` — tests for `recommend`, `smart_recommend`, edge cases
- [x] Write `test_semantic.py` — tests for `recommend_by_query`
- [x] Complete README — motivation, dataset, methodology, architecture, examples, limitations
- [x] Build **presentation slides** (12 slides covering full pipeline)
- [ ] Add demo screenshots to `demo_screenshots/` folder (at least 3 screenshots)
- [ ] Coordinate Q&A prep — ensure each member can answer questions on their section
- [ ] Be able to explain: *"What are the limitations of your system? What would you improve?"*

---

## Presentation Structure (15 min + demo + Q&A)

| Slide | Content | Who presents |
|-------|---------|--------------|
| 1 | Title + team intro | Eliott |
| 2 | Problem definition — why book recommendation is hard | Eliott |
| 3 | Dataset overview — size, columns, distributions, missing values | Sai Aditya Talluri |
| 4 | NLP pipeline overview — full flow diagram | Martina |
| 5 | Preprocessing — weighted combined_text + clean_text steps | Sai Aditya Talluri |
| 6 | TF-IDF explained — how vectorization works, why cosine similarity | Martina |
| 7 | Sentence-BERT explained — embeddings, semantic search | Martina |
| 8 | Smart routing — how title / author / query are differentiated | Martina |
| 9 | System architecture — layered diagram | Richard |
| 10 | Live demo | Richard |
| 11 | Limitations | Eliott |
| 12 | Future improvements | Eliott |

---

## Timeline

| Milestone | Target Date | Owner |
|-----------|-------------|-------|
| Core pipeline working end-to-end | Week before presentation | Sai Aditya Talluri + Martina |
| GUI polished, demo script ready | 3 days before | Richard |
| Tests passing, README complete | 3 days before | Eliott |
| Slides first draft | 2 days before | Eliott |
| Demo screenshots added | 2 days before | Richard |
| Full dress rehearsal (all 4 members) | 1 day before | All |
| Presentation day | Presentation date | All |

---

## Q&A Preparation

Each member should be able to answer these questions:

**On the NLP pipeline (Martina leads, all must know):**
- What is a bag of words? What is TF-IDF?
- Why cosine similarity and not Euclidean distance?
- What exactly happens between raw text and final recommendations?
- What is a word embedding? How does Sentence-BERT differ from TF-IDF?

**On the data (Sai Aditya Talluri leads):**
- Why did you choose those 4 columns and not others?
- Why is title weighted 3x and description 1x?
- What did you do about missing values?

**On the system (Richard leads):**
- How does the routing between TF-IDF and SBERT work?
- Why Streamlit? What would you use in production?
- What does the sidebar filtering do and how is it implemented?

**On limitations (Eliott leads):**
- What are the main weaknesses of your approach?
- What would a truly personalized recommender need that yours lacks?
- How would you scale this to millions of books?
