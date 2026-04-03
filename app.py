"""
Streamlit web interface for the book recommender system.

Professional dark-theme UI with:
  - Book cover images (thumbnail column)
  - Card grid layout (3 columns)
  - Sidebar filters: category, min rating, year range, result count
  - Smart routing badge (title / author / semantic)
  - Similarity score progress bar
  - Star rating display
"""

import sys
from pathlib import Path

# Allow running as: python src/app.py  OR  streamlit run src/app.py
if __name__ == "__main__" or __package__ is None:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import pandas as pd
import streamlit as st

from src.recommender import smart_recommend, recommender

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BookFinder — Discover Your Next Read",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global background ───────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0d1b2a 0%, #1a2e44 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: #080f1a;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #8ab4d4 !important;
}

/* ── Hero ────────────────────────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 2.2rem 0 0.8rem 0;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #b8942a, #f0d47a, #c9a84c, #f0d47a, #b8942a);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1.5px;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.hero-sub {
    color: #6a94b8;
    font-size: 1.05rem;
    margin-top: 0;
    margin-bottom: 0;
}

/* ── Search input ────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: #0f1e30 !important;
    border: 2px solid #1e3a5f !important;
    border-radius: 14px !important;
    color: #e8f4fd !important;
    font-size: 1.05rem !important;
    padding: 0.85rem 1.3rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 3px rgba(201,168,76,0.18) !important;
}
[data-testid="stTextInput"] input::placeholder { color: #3a5a78 !important; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #c9a84c 0%, #e8c966 100%) !important;
    color: #0a1420 !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.4rem !important;
    width: 100% !important;
    transition: all 0.18s !important;
    letter-spacing: 0.3px;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(201,168,76,0.45) !important;
}

/* ── Book card ───────────────────────────────────────────────────────────── */
.book-card {
    background: linear-gradient(160deg, #0f1e30 0%, #0a1628 100%);
    border: 1px solid #1a3350;
    border-radius: 16px;
    padding: 1rem 1rem 1.1rem 1rem;
    margin-bottom: 1.2rem;
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
    position: relative;
    overflow: hidden;
}
.book-card:hover {
    transform: translateY(-5px);
    border-color: #c9a84c;
    box-shadow: 0 10px 36px rgba(201,168,76,0.22);
}
.book-cover {
    width: 100%;
    height: 210px;
    object-fit: cover;
    border-radius: 10px;
    margin-bottom: 0.85rem;
    display: block;
    background: #0f1e30;
}
.book-cover-placeholder {
    width: 100%;
    height: 210px;
    background: linear-gradient(135deg, #132235, #0a1628);
    border-radius: 10px;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3.5rem;
}
.book-title {
    font-size: 0.97rem;
    font-weight: 700;
    color: #deeef8;
    line-height: 1.35;
    margin: 0 0 0.25rem 0;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.book-author {
    font-size: 0.82rem;
    color: #6a94b8;
    margin: 0 0 0.55rem 0;
    font-style: italic;
}
.category-badge {
    display: inline-block;
    background: rgba(201,168,76,0.12);
    color: #c9a84c;
    border: 1px solid rgba(201,168,76,0.28);
    border-radius: 20px;
    padding: 0.15rem 0.65rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2px;
    margin-bottom: 0.5rem;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    display: block;
}
.book-meta {
    font-size: 0.78rem;
    color: #4a7a9e;
    margin-bottom: 0.4rem;
}
.stars { color: #c9a84c; font-size: 0.88rem; letter-spacing: 1px; }
.score-label {
    font-size: 0.72rem;
    color: #4a7a9e;
    margin: 0.55rem 0 0.25rem 0;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.score-bar-bg {
    background: #0d1a28;
    border-radius: 4px;
    height: 5px;
    overflow: hidden;
    border: 1px solid #1a3350;
}
.score-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #8a6820, #c9a84c, #f0d47a);
    transition: width 0.4s ease;
}
.desc-text {
    font-size: 0.76rem;
    color: #3f6a8a;
    margin-top: 0.55rem;
    line-height: 1.45;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr { border-color: #1a3350 !important; margin: 0.8rem 0 !important; }

/* ── Results info bar ────────────────────────────────────────────────────── */
.results-bar {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.4rem;
    padding: 0.6rem 1rem;
    background: #0a1628;
    border: 1px solid #1a3350;
    border-radius: 10px;
}
.results-text { color: #6a94b8; font-size: 0.88rem; }
.mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.mode-title  { background: rgba(80,200,100,0.12); color: #50c864; border: 1px solid rgba(80,200,100,0.3); }
.mode-author { background: rgba(80,130,255,0.12); color: #6490ff; border: 1px solid rgba(80,130,255,0.3); }
.mode-query  { background: rgba(201,168,76,0.12); color: #c9a84c; border: 1px solid rgba(201,168,76,0.3); }

/* ── Tip cards ───────────────────────────────────────────────────────────── */
.tip-card {
    background: #0a1628;
    border: 1px solid #1a3350;
    border-radius: 14px;
    padding: 1.3rem;
    text-align: center;
    height: 100%;
}
.tip-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
.tip-title { color: #c9a84c; font-weight: 700; font-size: 0.95rem; margin-bottom: 0.4rem; }
.tip-desc { color: #3f6a8a; font-size: 0.8rem; line-height: 1.45; }

/* ── Example buttons ─────────────────────────────────────────────────────── */
.example-chip {
    background: #0a1628;
    border: 1px solid #1a3350;
    border-radius: 24px;
    padding: 0.5rem 1.1rem;
    color: #c9a84c;
    font-size: 0.85rem;
    font-weight: 600;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.15s;
}
.example-chip:hover { border-color: #c9a84c; }

/* ── Section label ───────────────────────────────────────────────────────── */
.section-label {
    color: #c9a84c;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
}

/* ── Sidebar labels ──────────────────────────────────────────────────────── */
.sb-label {
    color: #c9a84c !important;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    display: block;
    margin-top: 1rem;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def star_display(rating: float) -> str:
    """Convert a float rating to a unicode star string."""
    if rating is None:
        return ""
    full = int(rating)
    half = 1 if (rating - full) >= 0.3 else 0
    empty = 5 - full - half
    return "★" * full + ("½" if half else "") + "☆" * empty


def book_card_html(book: dict) -> str:
    """Render a single book as an HTML card."""
    title       = book.get("title", "Unknown Title")
    authors     = book.get("authors", "Unknown Author")
    thumbnail   = book.get("thumbnail", "")
    category    = book.get("categories", "")
    rating      = book.get("average_rating")
    year        = book.get("published_year")
    description = book.get("description", "")
    sim_score   = book.get("similarity_score")
    auth_rating = book.get("rating")

    # Cover image
    if thumbnail and str(thumbnail).strip().startswith("http"):
        cover = f'<img class="book-cover" src="{thumbnail}" onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\'" /><div class="book-cover-placeholder" style="display:none">📖</div>'
    else:
        cover = '<div class="book-cover-placeholder">📖</div>'

    # Category badge
    cat_html = ""
    if category and str(category).strip():
        cat_text = str(category).strip()[:45]
        cat_html = f'<span class="category-badge">{cat_text}</span>'

    # Meta line: year · rating
    meta_parts = []
    if year:
        meta_parts.append(str(int(year)))
    display_rating = rating or (auth_rating * 5 / 5 if auth_rating else None)
    if display_rating:
        meta_parts.append(f"⭐ {display_rating:.1f}")
    meta_html = f'<div class="book-meta">{" · ".join(meta_parts)}</div>' if meta_parts else ""

    # Stars
    stars = star_display(display_rating)
    stars_html = f'<div class="stars">{stars}</div>' if stars else ""

    # Score bar
    if sim_score is not None:
        pct = max(0, min(100, int(sim_score * 100)))
        score_html = f"""
        <div class="score-label">Match score</div>
        <div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%"></div></div>
        <div style="text-align:right;font-size:0.7rem;color:#4a7a9e;margin-top:0.2rem">{sim_score:.3f}</div>"""
    elif auth_rating is not None:
        pct = max(0, min(100, int((auth_rating / 5.0) * 100)))
        score_html = f"""
        <div class="score-label">Rating</div>
        <div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%"></div></div>
        <div style="text-align:right;font-size:0.7rem;color:#4a7a9e;margin-top:0.2rem">{auth_rating:.2f} / 5.0</div>"""
    else:
        score_html = ""

    # Description
    desc_html = ""
    if description and str(description).strip():
        desc_html = f'<div class="desc-text">{str(description).strip()[:220]}</div>'

    return f"""
    <div class="book-card">
        {cover}
        {cat_html}
        <div class="book-title">{title}</div>
        <div class="book-author">{authors}</div>
        {meta_html}
        {stars_html}
        {score_html}
        {desc_html}
    </div>"""


def detect_search_mode(user_input: str, results: list, df: pd.DataFrame) -> str:
    """Detect whether the result came from title, author, or semantic search."""
    if not results:
        return "query"
    if "rating" in results[0]:
        return "author"
    titles_lower = df["title"].astype(str).str.lower().str.strip()
    if user_input.lower().strip() in titles_lower.values:
        return "title"
    return "query"


def enrich_results(results: list, df: pd.DataFrame) -> list:
    """Add thumbnail, categories, published_year, description, average_rating to each result."""
    enriched = []
    title_index = {str(t).lower().strip(): i for i, t in enumerate(df["title"].astype(str))}
    for r in results:
        r = dict(r)
        key = str(r.get("title", "")).lower().strip()
        idx = title_index.get(key)
        if idx is not None:
            row = df.iloc[idx]
            r["thumbnail"]     = str(row["thumbnail"])     if pd.notna(row.get("thumbnail"))     else ""
            r["categories"]    = str(row["categories"])    if pd.notna(row.get("categories"))    else ""
            r["average_rating"]= float(row["average_rating"]) if pd.notna(row.get("average_rating")) else None
            r["published_year"]= int(row["published_year"])   if pd.notna(row.get("published_year")) else None
            r["description"]   = str(row["description"])   if pd.notna(row.get("description"))   else ""
        else:
            r.setdefault("thumbnail", "")
            r.setdefault("categories", "")
            r.setdefault("average_rating", None)
            r.setdefault("published_year", None)
            r.setdefault("description", "")
        enriched.append(r)
    return enriched


@st.cache_data
def load_filter_options():
    """Load category list and year range from the dataset (cached)."""
    df = recommender.get_data()
    cats = sorted(df["categories"].dropna().astype(str).unique().tolist())
    min_yr = int(df["published_year"].dropna().min()) if "published_year" in df.columns else 1900
    max_yr = int(df["published_year"].dropna().max()) if "published_year" in df.columns else 2025
    return cats, min_yr, max_yr


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.5rem;font-weight:900;color:#c9a84c;letter-spacing:-0.5px'>📚 BookFinder</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#3f6a8a;font-size:0.82rem;margin-bottom:0.5rem'>Powered by TF-IDF & Sentence-BERT</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown('<span class="sb-label">Results</span>', unsafe_allow_html=True)
    top_n = st.slider("Number of results", min_value=3, max_value=20, value=9, step=3,
                      label_visibility="collapsed")

    st.markdown('<span class="sb-label">Filters</span>', unsafe_allow_html=True)
    cats, min_yr, max_yr = load_filter_options()

    selected_cats = st.multiselect(
        "Categories",
        options=cats,
        placeholder="All categories",
        label_visibility="collapsed",
    )

    min_rating = st.slider("Min rating", 0.0, 5.0, 0.0, 0.5, format="⭐ %.1f",
                           label_visibility="collapsed")

    year_range = st.slider(
        "Year range",
        min_value=min_yr, max_value=max_yr,
        value=(min_yr, max_yr),
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<span class="sb-label">Search modes</span>', unsafe_allow_html=True)
    st.html("""
<div style="font-size:0.8rem;line-height:1.9;color:#3f6a8a">
📖 <b style="color:#50c864">Book title</b> → similar books<br>
✍️ <b style="color:#6490ff">Author name</b> → their top books<br>
🔍 <b style="color:#c9a84c">Topic / phrase</b> → AI semantic search
</div>""")


# ── Hero ──────────────────────────────────────────────────────────────────────
st.html("""
<div class="hero">
  <div class="hero-title">📚 BookFinder</div>
  <p class="hero-sub">Discover your next great read — by title, author, or any topic you can think of</p>
</div>""")

# ── Search bar ────────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        label="search",
        label_visibility="collapsed",
        placeholder="e.g.   Gilead   ·   Stephen King   ·   fantasy magic   ·   books about travel",
        key="search_input",
    )
with col_btn:
    search_clicked = st.button("Search 🔍")

st.markdown("<div style='margin-bottom:0.4rem'></div>", unsafe_allow_html=True)
st.markdown("---")


# ── Search logic ──────────────────────────────────────────────────────────────
if search_clicked and user_input.strip():
    with st.spinner("Finding your next read..."):
        # Fetch generous pool so filters still leave enough results
        pool = max(top_n * 6, 60)
        raw = recommender.smart_recommend(user_input.strip(), top_n=pool)

    if not raw:
        st.warning("No results found. Try a different query.")
    else:
        df_full = recommender.get_data()
        results = enrich_results(raw, df_full)

        # Apply sidebar filters
        if selected_cats:
            results = [b for b in results
                       if any(c.lower() in b.get("categories", "").lower() for c in selected_cats)]
        if min_rating > 0:
            results = [b for b in results if (b.get("average_rating") or 0) >= min_rating]
        if year_range != (min_yr, max_yr):
            results = [b for b in results
                       if b.get("published_year") and year_range[0] <= b["published_year"] <= year_range[1]]

        display = results[:top_n]

        if not display:
            st.warning("No books match your current filters. Try relaxing the category, rating, or year filters.")
        else:
            mode = detect_search_mode(user_input.strip(), display, df_full)
            mode_map = {
                "title":  ("mode-title",  "📖 Title match"),
                "author": ("mode-author", "✍️ Author search"),
                "query":  ("mode-query",  "🔍 Semantic search"),
            }
            badge_cls, badge_label = mode_map.get(mode, ("mode-query", "🔍 Semantic search"))

            st.html(f"""
            <div class="results-bar">
              <span class="results-text">
                Showing <strong style="color:#deeef8">{len(display)}</strong> results for
                <strong style="color:#deeef8">"{user_input}"</strong>
              </span>
              <span class="mode-badge {badge_cls}">{badge_label}</span>
            </div>""")

            # Card grid — 3 per row
            cols_per_row = 3
            for row_i in range(0, len(display), cols_per_row):
                row_books = display[row_i: row_i + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, book in zip(cols, row_books):
                    with col:
                        st.html(book_card_html(book))

elif search_clicked:
    st.warning("Please enter a search term.")

else:
    # ── Landing page ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Try these examples</div>', unsafe_allow_html=True)
    ex_cols = st.columns(4)
    examples = [
        ("📖", "Gilead"),
        ("✍️", "Agatha Christie"),
        ("🔮", "fantasy magic"),
        ("🌍", "books about travel"),
    ]
    for col, (icon, ex) in zip(ex_cols, examples):
        with col:
            st.html(f"""
            <div class="example-chip">{icon} {ex}</div>
            """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)

    tip_cols = st.columns(3)
    tips = [
        ("📖", "Book Title",
         "Type any exact book title. The system uses <b>TF-IDF cosine similarity</b> to find books "
         "with the most overlapping vocabulary across title, category, author, and description."),
        ("✍️", "Author Name",
         "Type an author's name to browse <b>their catalogue</b>, sorted by average reader rating "
         "from highest to lowest."),
        ("🔍", "Topic or Theme",
         "Type anything — <i>\"dark academia\"</i>, <i>\"survival in the wild\"</i>, <i>\"romance and love\"</i>. "
         "<b>Sentence-BERT</b> encodes meaning and finds semantically similar books."),
    ]
    for col, (icon, title, desc) in zip(tip_cols, tips):
        with col:
            st.html(f"""
            <div class="tip-card">
              <div class="tip-icon">{icon}</div>
              <div class="tip-title">{title}</div>
              <div class="tip-desc">{desc}</div>
            </div>""")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Dataset at a glance</div>', unsafe_allow_html=True)

    stat_cols = st.columns(4)
    stats = [
        ("6,810", "Books indexed"),
        ("3,780", "Unique authors"),
        ("567",   "Categories"),
        ("all-MiniLM-L6-v2", "Embedding model"),
    ]
    for col, (val, label) in zip(stat_cols, stats):
        with col:
            st.html(f"""
            <div style="background:#0a1628;border:1px solid #1a3350;border-radius:12px;
                        padding:1rem;text-align:center;">
              <div style="font-size:1.5rem;font-weight:800;color:#c9a84c">{val}</div>
              <div style="font-size:0.78rem;color:#3f6a8a;margin-top:0.2rem">{label}</div>
            </div>""")
