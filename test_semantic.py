"""
Quick test for semantic search: recommend_by_query("books about travelling", 5).
Run from project root: python test_semantic.py
"""

from src.recommender import recommender

if __name__ == "__main__":
    query = "books about travelling"
    top_n = 5
    print(f"recommend_by_query('{query}', {top_n}):\n")
    recs = recommender.recommend_by_query(query, top_n=top_n)
    for i, r in enumerate(recs, start=1):
        print(f"{i}. {r['title']}")
        print(f"   Authors: {r['authors']}")
        print(f"   Similarity: {r['similarity_score']:.4f}\n")
    if not recs:
        print("No recommendations returned.")
