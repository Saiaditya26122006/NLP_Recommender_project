"""
Test script: load dataset using preprocessing.py and print first 5 rows of combined_text.
Run from project root: python test_preprocessing.py
"""

from src.preprocessing import load_data, preprocess_data, COMBINED_COLUMN

if __name__ == "__main__":
    # Load and preprocess
    df = load_data()
    df = preprocess_data(df)

    # Print first 5 rows of combined_text
    print("First 5 rows of combined_text:\n")
    for i, text in enumerate(df[COMBINED_COLUMN].head(5), start=1):
        print(f"--- Row {i} ---")
        # Truncate long text for readability
        display = text if len(text) <= 500 else text[:500] + "..."
        print(display)
        print()
