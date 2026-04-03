"""
Test script: import recommender and print recommendations for "Gilead".
Run from project root: python test_recommender.py
"""

from src.recommender import recommend

if __name__ == "__main__":
    book_title = "Gilead"
    top_n = 5
    print(f"Recommendations for '{book_title}':\n")
    try:
        recs = recommend(book_title, top_n=top_n)
        for i, r in enumerate(recs, start=1):
            print(f"{i}. {r['title']}")
            print(f"   Authors: {r['authors']}")
            print(f"   Similarity: {r['similarity_score']:.4f}\n")
        if not recs:
            print("No recommendations found.")
    except ValueError as e:
        print(f"Error: {e}")
