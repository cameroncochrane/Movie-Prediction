import ast
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MOVIES_PATH = "data/tmdb_5000_movies.csv"
CREDITS_PATH = "data/tmdb_5000_credits.csv"


def _safe_literal_eval(x):
    """
    The TMDB CSV stores JSON-like lists as strings.
    Example: '[{"id": 28, "name": "Action"}, ...]'
    We parse them into Python objects safely using ast.literal_eval.
    If anything fails, return an empty list.
    """
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


def _extract_names(list_of_dicts, top_n=None):
    """
    Convert a list of dicts like [{"name":"Action"}, {"name":"Drama"}]
    into a list of names like ["Action", "Drama"].
    Optionally keep only the first top_n.
    """
    names = []
    for d in list_of_dicts:
        if isinstance(d, dict):
            n = d.get("name")
            if isinstance(n, str) and n.strip():
                names.append(n.strip())
    return names[:top_n] if top_n else names


def _get_director(crew_list):
    """
    crew_list is a list of dicts; find the crew member with job == 'Director'.
    If not found, return empty string.
    """
    for d in crew_list:
        if isinstance(d, dict) and d.get("job") == "Director":
            n = d.get("name")
            return n if isinstance(n, str) else ""
    return ""


def load_catalogue_with_soup():
    """
    Main loader:
    - Reads both CSV files
    - Merges movies + credits
    - Extracts tokens for content-based similarity
    - Builds "soup" column (one big string per movie)
    """
    movies = pd.read_csv(MOVIES_PATH)
    credits = pd.read_csv(CREDITS_PATH, low_memory=False)

    # credits.csv sometimes has lots of empty "Unnamed:*" columns; drop them
    credits = credits.loc[:, ~credits.columns.str.startswith("Unnamed:")]

    # Ensure IDs are ints so join works reliably
    movies["id"] = movies["id"].astype(int)
    credits["movie_id"] = pd.to_numeric(credits["movie_id"], errors="coerce")
    credits = credits.dropna(subset=["movie_id"])
    credits["movie_id"] = credits["movie_id"].astype(int)

    # Merge: movies.id == credits.movie_id
    df = movies.merge(credits, left_on="id", right_on="movie_id", how="left")
    df["title"] = movies["title"]

    # Parse JSON-like strings into Python lists
    df["genres"] = df["genres"].apply(_safe_literal_eval)
    df["keywords"] = df["keywords"].apply(_safe_literal_eval)
    df["cast"] = df["cast"].apply(_safe_literal_eval)
    df["crew"] = df["crew"].apply(_safe_literal_eval)

    # Extract "names" from these lists
    df["genre_names"] = df["genres"].apply(lambda x: _extract_names(x))
    df["keyword_names"] = df["keywords"].apply(lambda x: _extract_names(x))
    df["cast_names"] = df["cast"].apply(lambda x: _extract_names(x, top_n=5))  # top 5 cast
    df["director"] = df["crew"].apply(_get_director)

    # Overview is plain text; fill missing
    df["overview"] = df["overview"].fillna("").astype(str)

    # Normalize tokens: remove spaces and lower-case
    # e.g. "Sam Worthington" -> "samworthington" so it becomes a single token
    def norm(tokens):
        return [t.replace(" ", "").lower() for t in tokens if isinstance(t, str)]

    df["genre_names"] = df["genre_names"].apply(norm)
    df["keyword_names"] = df["keyword_names"].apply(norm)
    df["cast_names"] = df["cast_names"].apply(norm)
    df["director"] = df["director"].fillna("").astype(str).str.replace(" ", "").str.lower()

    # Build the soup: one big string per movie.
    # You can tweak weighting by duplicating certain parts (e.g. genres twice).
    df["soup"] = (
        df["genre_names"].apply(lambda x: " ".join(x)) + " " +
        df["keyword_names"].apply(lambda x: " ".join(x)) + " " +
        df["cast_names"].apply(lambda x: " ".join(x)) + " " +
        df["director"] + " " +
        df["overview"].str.lower()
    ).str.strip()

    # Keep only what we need for UI + recs
    df["title"] = df["title"].fillna("").astype(str)
    df["release_date"] = df.get("release_date", "").fillna("").astype(str)

    out = df[[
        "id", "title", "release_date",
        "vote_average", "vote_count",
        "overview", "soup"
    ]].copy()

    # Remove duplicates just in case
    out = out.drop_duplicates(subset=["id"]).reset_index(drop=True)

    return out

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Content Recommender
class ContentRecommender:
    def __init__(self, df):
        """
        df: DataFrame containing at least columns: ['id', 'soup']
        """
        self.df = df.reset_index(drop=True)

        # TF-IDF converts text -> numeric vectors
        # stop_words="english" removes common words that don't help similarity much
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2)  # unigrams + bigrams
        )

        # Matrix shape: (n_movies, n_features)
        self.tfidf = self.vectorizer.fit_transform(self.df["soup"].fillna(""))

        # Map movie_id -> row index, for quick lookup
        self.id_to_idx = {int(mid): i for i, mid in enumerate(self.df["id"].tolist())}

    def recommend_by_id(self, movie_id: int, top_n: int = 10):
        """
        Given a movie_id, return top_n similar movies.

        Steps:
        1) locate that movie's TF-IDF vector
        2) compute cosine similarity vs all other movies
        3) take the highest scores (excluding itself)
        """
        movie_id = int(movie_id)
        if movie_id not in self.id_to_idx:
            return []

        idx = self.id_to_idx[movie_id]

        # cosine_similarity between one vector and the full matrix
        # result is (1, n_movies) -> flatten to (n_movies,)
        sims = cosine_similarity(self.tfidf[idx], self.tfidf).flatten()

        # Exclude the movie itself
        sims[idx] = -1.0

        # Sort indices by similarity descending and take top_n
        top_idx = np.argsort(sims)[::-1][:top_n]

        recs = self.df.iloc[top_idx].copy()
        recs["similarity"] = sims[top_idx]
        return recs.to_dict("records")