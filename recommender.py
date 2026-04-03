"""
Book recommender engine: TF-IDF + Sentence-BERT embeddings.

- recommend(book_title): similar books (TF-IDF cosine similarity).
- recommend_by_query(query): semantic search via Sentence-BERT embeddings.
- recommend_by_author(author_name): books by author.
- smart_recommend(user_input): routes to title / author / query.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

from src.preprocessing import load_data, preprocess_data, clean_text

# Return value when the requested book is not found (safe, no crash)
BOOK_NOT_FOUND_RESULT: List[dict] = [
    {"Title": "Book not found", "Authors": "", "Similarity Score": ""}
]


class BookRecommender:
    """
    Book recommendation engine: TF-IDF for title-based similarity and
    Sentence-BERT for query-based semantic search. All embeddings and
    matrices are computed once in __init__ and not recomputed on each call.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        """
        Load dataset, preprocess, build TF-IDF and book-to-book similarity,
        load Sentence-BERT and compute book embeddings (once).
        """
        df = load_data(path)
        self._df = preprocess_data(df)

        # TF-IDF and book-to-book similarity (for recommend by title)
        self._vectorizer = TfidfVectorizer(
            max_features=10_000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(
            self._df["combined_text"].fillna("")
        )
        self._similarity_matrix = cosine_similarity(
            self._tfidf_matrix, self._tfidf_matrix
        )

        # Sentence-BERT: one-time encoding of all books (for recommend_by_query)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = self._df["combined_text"].fillna("").tolist()
        self.book_embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def _find_book_index(self, title: str) -> Optional[int]:
        """
        Return the row index for the given book title, or None if not found.
        Matching is case-insensitive with stripped whitespace.
        """
        if title is None:
            return None
        title_clean = str(title).strip()
        if not title_clean:
            return None

        titles = self._df["title"].astype(str).str.strip().str.lower()
        target = title_clean.lower()
        mask = titles == target
        indices = self._df.index[mask].tolist()
        if indices:
            return int(indices[0])
        return None

    def recommend(self, book_title: str, top_n: int = 5) -> List[dict]:
        """
        Return top_n books most similar to the given book by title.

        Args:
            book_title: Title of the book (case-insensitive). Can be None or empty.
            top_n: Number of recommendations (default 5). Clamped to a valid range.

        Returns:
            List of dicts with keys: title, authors, similarity_score. Sorted by
            similarity descending. Returns BOOK_NOT_FOUND_RESULT if book not found
            or input is invalid.
        """
        try:
            if book_title is None:
                return BOOK_NOT_FOUND_RESULT
            title_str = str(book_title).strip()
            if not title_str:
                return BOOK_NOT_FOUND_RESULT

            idx = self._find_book_index(title_str)
            if idx is None:
                return BOOK_NOT_FOUND_RESULT

            top_n = max(0, int(top_n)) if isinstance(top_n, (int, float)) else 5
            n = min(top_n, len(self._similarity_matrix) - 1)
            if n <= 0:
                return []

            scores = self._similarity_matrix[idx].copy()
            scores[idx] = -1.0  # Exclude the input book

            top_indices = scores.argsort()[::-1][:n]
            result = []
            for i in top_indices:
                if i == idx:
                    continue
                row = self._df.iloc[i]
                result.append({
                    "title": row["title"],
                    "authors": row["authors"],
                    "similarity_score": float(scores[i]),
                })
            return result
        except Exception:
            return BOOK_NOT_FOUND_RESULT

    def recommend_by_query(self, query: str, top_n: int = 5) -> List[dict]:
        """
        Return top_n books most similar to a free-text query using Sentence-BERT.

        Cleans the query, encodes it with the same embedding model as the corpus,
        computes cosine similarity to precomputed book_embeddings (no recompute).
        Returns title, authors, similarity_score.
        """
        if query is None or not str(query).strip():
            return []
        cleaned = clean_text(str(query).strip())
        if not cleaned:
            return []
        top_n = max(0, int(top_n)) if isinstance(top_n, (int, float)) else 5
        n = min(top_n, len(self._df))
        if n <= 0:
            return []
        query_embedding = self.embedding_model.encode(
            [cleaned], convert_to_numpy=True
        )
        similarities = cosine_similarity(
            query_embedding, self.book_embeddings
        )[0]
        top_indices = similarities.argsort()[::-1][:n]
        result = []
        for i in top_indices:
            row = self._df.iloc[i]
            result.append({
                "title": row["title"],
                "authors": row["authors"],
                "similarity_score": float(similarities[i]),
            })
        return result

    def recommend_by_author(self, author_name: str, top_n: int = 5) -> List[dict]:
        """
        Return up to top_n books by the given author (case-insensitive match on authors).

        Matches author_name inside the authors column, optionally sorts by average_rating
        if that column exists, and returns title, authors, and rating.
        """
        if author_name is None or not str(author_name).strip():
            return []
        name = str(author_name).strip().lower()
        top_n = max(0, int(top_n)) if isinstance(top_n, (int, float)) else 5
        if top_n <= 0:
            return []

        authors_lower = self._df["authors"].astype(str).str.lower()
        mask = authors_lower.str.contains(name, regex=False)
        matches = self._df.loc[mask]

        if matches.empty:
            return []

        if "average_rating" in matches.columns:
            matches = matches.sort_values(
                "average_rating", ascending=False, na_position="last"
            )
        matches = matches.head(top_n)

        result = []
        for _, row in matches.iterrows():
            rating = row.get("average_rating")
            if rating is not None and (isinstance(rating, (int, float)) or (isinstance(rating, str) and rating.strip())):
                try:
                    rating = float(rating)
                except (TypeError, ValueError):
                    rating = None
            else:
                rating = None
            result.append({
                "title": row["title"],
                "authors": row["authors"],
                "rating": rating,
            })
        return result

    def smart_recommend(self, user_input: str, top_n: int = 5) -> List[dict]:
        """
        Route user_input to the best recommender: title match → recommend(),
        author match → recommend_by_author(), else → recommend_by_query().

        Phrase-like input (e.g. "romance and love", "books about travel") is
        never treated as a book title; it goes to author match or query search.
        """
        if user_input is None or not str(user_input).strip():
            return []
        user_input = str(user_input).strip()
        # Don't treat topic phrases as book titles: use query path instead
        looks_like_topic = (
            " and " in user_input
            or " about " in user_input
            or user_input.lower().startswith("books ")
        )
        if not looks_like_topic and self._find_book_index(user_input) is not None:
            return self.recommend(user_input, top_n)
        authors_lower = self._df["authors"].astype(str).str.lower()
        if authors_lower.str.contains(user_input.lower(), regex=False).any():
            return self.recommend_by_author(user_input, top_n)
        return self.recommend_by_query(user_input, top_n)

    def get_data(self) -> pd.DataFrame:
        """Return a copy of the preprocessed dataset (with combined_text)."""
        return self._df.copy()


# Global instance: built once at import; TF-IDF, similarity matrix, and book
# embeddings are not recomputed on each recommend() / recommend_by_query()
recommender = BookRecommender()


def recommend(book_title: str, top_n: int = 5) -> List[dict]:
    """
    Return top_n book recommendations for the given title (module-level API).

    Delegates to the global BookRecommender instance. TF-IDF and similarity
    are computed once when the module is loaded, not on every call.
    """
    return recommender.recommend(book_title, top_n)


def smart_recommend(user_input: str, top_n: int = 5) -> List[dict]:
    """
    Return recommendations for any user input (module-level API).

    Routes to title-based, author-based, or query-based (semantic) recommendations.
    Delegates to the global BookRecommender instance.
    """
    return recommender.smart_recommend(user_input, top_n)


def get_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Return a copy of the preprocessed dataset from the global recommender.
    The path argument is ignored after the recommender is created.
    """
    return recommender.get_data()


if __name__ == "__main__":
    import sys
    title = sys.argv[1] if len(sys.argv) > 1 else None
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    if not title or not str(title).strip():
        print("Usage: python -m src.recommender <book_title> [top_n]")
        sys.exit(1)
    recs = recommend(title, top_n=top_n)
    if recs == BOOK_NOT_FOUND_RESULT:
        print(recs[0]["Title"])
        sys.exit(1)
    print(f"Top {len(recs)} recommendations for '{title}':\n")
    for r in recs:
        print(f"  {r['title']} | {r['authors']} | score={r['similarity_score']:.4f}")
