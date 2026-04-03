"""
NLP preprocessing module for the book recommender.

Loads book data from data/books.csv, combines title/authors/categories/description
into one column (combined_text), cleans it, and returns the dataframe for downstream NLP.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import re

# NLTK stopwords are loaded once and cached
_nltk_stopwords = None

# Columns to combine into one text field
TEXT_COLUMNS = ("title", "authors", "categories", "description")
COMBINED_COLUMN = "combined_text"

# Weights for combined_text: title and categories count more for recommendation quality
# combined_text = title*3 + categories*2 + authors + description
TEXT_WEIGHTS = (
    ("title", 3),
    ("categories", 2),
    ("authors", 1),
    ("description", 1),
)

# Default path to the dataset (relative to project root)
DEFAULT_DATA_PATH = "data/books.csv"


def _get_stopwords() -> set:
    """Load NLTK English stopwords, downloading the corpus if needed."""
    global _nltk_stopwords
    if _nltk_stopwords is None:
        try:
            import nltk
            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            _nltk_stopwords = set(stopwords.words("english"))
        except Exception as e:
            raise ImportError(
                "NLTK stopwords could not be loaded. Install with: pip install nltk"
            ) from e
    return _nltk_stopwords


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the book dataset from a CSV file.

    Args:
        path: Path to the CSV file. If None, uses data/books.csv relative to
              the project root (parent of src/).

    Returns:
        DataFrame with book records.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns (title, authors, categories, description) are missing.
    """
    if path is None:
        # Resolve project root (parent of src/) and use data/books.csv
        src_dir = Path(__file__).resolve().parent
        project_root = src_dir.parent
        path = str(project_root / DEFAULT_DATA_PATH)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required = {"title", "authors", "categories", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset must contain columns: {required}. Missing: {missing}. "
            f"Found: {list(df.columns)}"
        )

    return df


def clean_text(text: str) -> str:
    """
    Clean a single text string for NLP.

    - Converts to lowercase
    - Removes punctuation
    - Removes numbers
    - Removes NLTK English stopwords
    - Removes extra whitespace

    Args:
        text: Raw input text.

    Returns:
        Cleaned text as a single string (space-joined words).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation: keep only letters, digits, and spaces (digits removed next)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove numbers (strip digits)
    text = re.sub(r"\d+", " ", text)

    # Remove extra whitespace: collapse multiple spaces and trim
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    if not tokens:
        return ""

    # Remove NLTK stopwords
    stopwords_set = _get_stopwords()
    tokens = [t for t in tokens if t not in stopwords_set]

    return " ".join(tokens).strip()


def _weighted_combined_row(row: pd.Series) -> str:
    """Build weighted combined text for one row: title*3 + categories*2 + authors + description."""
    parts = []
    for col, weight in TEXT_WEIGHTS:
        if col not in row.index:
            continue
        val = str(row[col]).strip() if pd.notna(row[col]) else ""
        if val:
            parts.append(" ".join([val] * weight))
    return " ".join(parts)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine title, authors, categories, and description into combined_text with weights
    (title x3, categories x2, authors x1, description x1), then apply clean_text.

    Args:
        df: DataFrame with at least columns: title, authors, categories, description.

    Returns:
        DataFrame with all original columns plus combined_text (cleaned).
    """
    available = [c for c, _ in TEXT_WEIGHTS if c in df.columns]
    if not available:
        raise ValueError(
            f"DataFrame must contain at least one of {[c for c, _ in TEXT_WEIGHTS]}. "
            f"Found columns: {list(df.columns)}"
        )

    # Build weighted combined text: title*3 + categories*2 + authors + description
    combined = df.apply(_weighted_combined_row, axis=1)
    combined = combined.str.replace(r"\s+", " ", regex=True).str.strip()

    df = df.copy()
    df["combined_text"] = combined

    # Apply clean_text so combined_text is fully cleaned
    df["combined_text"] = df["combined_text"].apply(clean_text)

    return df


if __name__ == "__main__":
    # Example: load from data/books.csv, preprocess, and print summary
    try:
        df = load_data()
        df = preprocess_data(df)
        print(f"Preprocessed {len(df)} books.")
        print("Columns:", list(df.columns))
        if len(df) > 0:
            sample = df[COMBINED_COLUMN].iloc[0]
            print("Sample combined_text (first row):", sample[:200] + ("..." if len(sample) > 200 else ""))
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
