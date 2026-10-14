"""
Hybrid video game recommender for Steam.

Dataset: "Game Recommendations on Steam" (Kaggle, antonkozyriev)
    data/games.csv            -> app_id, title, date_release, rating, positive_ratio, user_reviews, price_final...
    data/users.csv            -> user_id, products, reviews
    data/recommendations.csv  -> app_id, helpful, funny, date, is_recommended, hours, user_id, review_id
    data/games_metadata.json  -> app_id, description, tags

How the model works:

  1. HYBRID MODEL: two sources of information are combined
       - Collaborative: "people who liked X also liked Y" (item-item similarity between games).
       - Content: thematic similarity based on Steam tags (TF-IDF + cosine similarity).
  2. IMPLICIT FEEDBACK: instead of predicting a rating, a positive review counts as 1.
     Games the user has already reviewed (positively or negatively) are never recommended.
  3. POPULARITY PENALTY: very popular games get a small score penalty so the same blockbusters
     (CS2, GTA V, Terraria...) do not show up for everyone.
  4. EVALUATION WITH PRECISION@K: we check whether the games a user recommended in the test set
     appear in their top-k (plus recall@k, catalog coverage, popularity and thematic affinity).

This module holds all the logic, so main.py (console) and app.py (Streamlit) can reuse it.
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ---------------- Configuration ----------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_PATH = os.path.join(DATA_DIR, "filtered_reviews.pkl")  # Filtered reviews (makes later runs much faster)

MIN_USER_REVIEWS = 10      # Minimum number of reviews for a user to enter the model
MIN_GAME_REVIEWS = 50      # Minimum number of reviews for a game to enter the model
MAX_USERS = 100_000        # Random sample of users to keep memory under control. None = no limit
SEED = 42

NEIGHBORS = 100            # Number of most similar games stored for each game (sparse similarity matrix)
CONTENT_WEIGHT = 1.2       # Weight of the thematic part (tags) relative to the collaborative part (weight 1)
POPULARITY_WEIGHT = 0.3    # How much very popular games are penalized
SVD_FACTORS = 100          # Only used to compare against an SVD model during evaluation

IMAGE_URL = "https://cdn.akamai.steamstatic.com/steam/apps/{}/header.jpg"  # Steam header image from the app_id


# ---------------- Data loading and cleaning ----------------

def load_games():
    games = pd.read_csv(os.path.join(DATA_DIR, "games.csv"), low_memory=False)
    games = games.drop_duplicates(subset=["app_id"])

    # Keep only the useful columns and rename them for readability:
    games = games[["app_id", "title", "date_release", "rating", "positive_ratio", "user_reviews", "price_final"]]
    games = games.rename(columns={"rating": "steam_rating", "price_final": "price"})

    games["year"] = pd.to_datetime(games["date_release"], errors="coerce").dt.year.astype("Int64")
    games["image"] = games["app_id"].map(IMAGE_URL.format)

    # Steam tags (sorted from most to least voted by players):
    metadata_path = os.path.join(DATA_DIR, "games_metadata.json")
    if os.path.exists(metadata_path):
        metadata = pd.read_json(metadata_path, lines=True)[["app_id", "tags"]]
        games = games.merge(metadata, on="app_id", how="left")
    else:
        print("WARNING: data/games_metadata.json not found. The content part of the model will not work.")
        games["tags"] = None

    games["tag_list"] = games["tags"].apply(lambda t: t if isinstance(t, list) else [])
    games["tags"] = games["tag_list"].apply(lambda t: ", ".join(t[:4]))  # Top 4 tags, for display

    return games.drop(columns=["date_release"])


def load_users():
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"),
                        dtype={"user_id": "int32", "products": "int32", "reviews": "int32"})
    users = users.drop_duplicates(subset=["user_id"])
    return users.rename(columns={"products": "library_size", "reviews": "review_count"})


def _read_recommendations():
    """Reads recommendations.csv (~41M rows) in chunks and only with the columns we need."""
    dtypes = {"app_id": "int32", "user_id": "int32", "is_recommended": "bool", "hours": "float32"}
    chunks = pd.read_csv(os.path.join(DATA_DIR, "recommendations.csv"),
                         usecols=list(dtypes.keys()), dtype=dtypes, chunksize=5_000_000)
    return pd.concat(chunks, ignore_index=True)


def load_interactions(use_cache=True):
    """Returns the filtered reviews: user_id, app_id, hours, is_recommended."""
    columns = ["user_id", "app_id", "hours", "is_recommended"]
    if use_cache and os.path.exists(CACHE_PATH):
        return pd.read_pickle(CACHE_PATH)[columns]

    reviews = _read_recommendations()
    reviews = reviews.dropna().drop_duplicates(subset=["user_id", "app_id"])  # One review per user and game

    # Remove users and games with too few reviews:
    user_counts = reviews["user_id"].value_counts()
    reviews = reviews[reviews["user_id"].isin(user_counts[user_counts >= MIN_USER_REVIEWS].index)]

    game_counts = reviews["app_id"].value_counts()
    reviews = reviews[reviews["app_id"].isin(game_counts[game_counts >= MIN_GAME_REVIEWS].index)]

    # Sample users so everything fits in memory on a regular computer:
    if MAX_USERS is not None and reviews["user_id"].nunique() > MAX_USERS:
        sample = pd.Series(reviews["user_id"].unique()).sample(MAX_USERS, random_state=SEED)
        reviews = reviews[reviews["user_id"].isin(sample)]

    reviews = reviews[columns].reset_index(drop=True)
    reviews.to_pickle(CACHE_PATH)
    return reviews


# ---------------- Helpers ----------------

def _top_k_similarity(A, k, block_size=2000):
    """
    Cosine similarity between the rows of A (already normalized), keeping only the k most similar items per row.
    Computed in blocks to avoid building a dense games x games matrix all at once (14,000² ≈ 800 MB).
    """
    n = A.shape[0]
    k = min(k, n - 1)
    rows, cols, values = [], [], []
    for start in range(0, n, block_size):
        S = A[start:start + block_size] @ A.T
        S = S.toarray() if hasattr(S, "toarray") else np.asarray(S)
        S[np.arange(S.shape[0]), np.arange(start, start + S.shape[0])] = 0  # A game is not its own neighbor
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        rows.append(np.repeat(np.arange(start, start + S.shape[0]), k))
        cols.append(idx.ravel())
        values.append(np.take_along_axis(S, idx, axis=1).ravel())
    M = csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    M.eliminate_zeros()
    return M.astype("float32")


def _zscore(S):
    """Standardizes each row (mean 0, std 1) so scores on different scales can be added together."""
    return (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-9)


def _tag_matrix(tag_lists):
    """
    TF-IDF of tags. Steam tags are sorted by votes, so earlier tags get more weight (1/sqrt(position+1)),
    and everything is multiplied by the IDF so very common tags ("Indie", "Singleplayer") count less than
    specific ones ("Roguelike", "Survival Horror").
    """
    vocabulary = {t: i for i, t in enumerate(sorted({t for tags in tag_lists for t in tags}))}
    rows, cols, values = [], [], []
    for i, tags in enumerate(tag_lists):
        for position, tag in enumerate(tags):
            rows.append(i)
            cols.append(vocabulary[tag])
            values.append(1 / np.sqrt(position + 1))
    n = len(tag_lists)
    C = csr_matrix((values, (rows, cols)), shape=(n, max(len(vocabulary), 1)), dtype="float32")
    idf = np.log(n / (1 + np.asarray((C > 0).sum(axis=0)).ravel())) + 1
    return normalize(csr_matrix(C.multiply(idf))).astype("float32"), vocabulary


# ---------------- Model ----------------

class HybridRecommender:

    def __init__(self, interactions, games, neighbors=NEIGHBORS,
                 content_weight=CONTENT_WEIGHT, popularity_weight=POPULARITY_WEIGHT):
        self.interactions = interactions
        self.games = games.set_index("app_id")
        self.neighbors = neighbors
        self.content_weight = content_weight
        self.popularity_weight = popularity_weight

    # ---------- Training ----------
    def fit(self):
        # Train / validation / test split (70% / 15% / 15%)
        self.train, temp = train_test_split(self.interactions, test_size=0.3, random_state=SEED)
        self.validation, self.test = train_test_split(temp, test_size=0.5, random_state=SEED)

        # User and game indices (dictionaries for fast lookups)
        self.users = np.sort(self.train["user_id"].unique())
        self.game_ids = np.sort(self.interactions["app_id"].unique())
        self.user_index = {u: i for i, u in enumerate(self.users)}
        self.game_index = {g: i for i, g in enumerate(self.game_ids)}

        # (2) Implicit feedback: user x game matrix with a 1 if the user reviewed the game positively
        self.X_liked = self._build_matrix(self.train[self.train["is_recommended"]])
        self.X_seen = self._build_matrix(self.train)  # Everything reviewed (so it is never recommended)
        self.known_users = self.users[self.X_liked.getnnz(axis=1) > 0]  # Users with at least one liked game

        # (1a) Collaborative part: cosine similarity between games based on which users recommend them
        games_by_users = normalize(self.X_liked.T.tocsr())
        self.collab_similarity = _top_k_similarity(games_by_users, self.neighbors)
        self._games_by_users = games_by_users

        # (1b) Content part: cosine similarity between game tags
        tag_lists = self.games["tag_list"].reindex(self.game_ids)
        tag_lists = [t if isinstance(t, list) else [] for t in tag_lists]
        self.tag_matrix, self.tag_vocabulary = _tag_matrix(tag_lists)
        self._estimate_missing_tags()
        self.content_similarity = _top_k_similarity(self.tag_matrix, self.neighbors)

        # (3) Popularity: number of reviews per game (log scale, standardized)
        self.popularity = np.asarray(self.X_seen.sum(axis=0)).ravel()
        log_popularity = np.log1p(self.popularity)
        self.popularity_z = ((log_popularity - log_popularity.mean()) / log_popularity.std()).astype("float32")

        # Global top: out of the 100 most reviewed games, the ones with the highest share of positive reviews
        summary = self.interactions.groupby("app_id")["is_recommended"].agg(["count", "mean"])
        self.global_top = summary.nlargest(100, "count").sort_values("mean", ascending=False).index.tolist()

        # Games each user reviewed in the WHOLE dataset (so we never recommend something they already know)
        self.played = self.interactions.groupby("user_id")["app_id"].apply(set).to_dict()
        return self

    def _estimate_missing_tags(self):
        """
        Data cleaning: in games_metadata.json many VERY popular games (Portal 2, GTA V, Cyberpunk...) have an
        empty tag list. To keep them in the content part, their tags are estimated as the weighted average of
        the tags of their collaborative neighbors that do have tags.
        """
        has_tags = np.asarray(self.tag_matrix.getnnz(axis=1) > 0)
        no_tags = ~has_tags
        # Keep only similarities towards tagged neighbors, and only for the rows of untagged games
        neighbors = diags(no_tags.astype("float32")) @ self.collab_similarity @ diags(has_tags.astype("float32"))
        estimated = normalize(neighbors @ self.tag_matrix).astype("float32")
        self.tag_matrix = csr_matrix(self.tag_matrix + estimated)

        # Store the top 4 estimated tags of each game for display
        tag_names = np.array(sorted(self.tag_vocabulary, key=self.tag_vocabulary.get))
        self.estimated_tags = {}
        for i in np.where(no_tags)[0]:
            row = estimated[i]
            if row.nnz:
                order = row.indices[np.argsort(-row.data)][:4]
                self.estimated_tags[self.game_ids[i]] = ", ".join(tag_names[order])
        self.n_estimated_tags = int(no_tags.sum())

    def _build_matrix(self, df):
        df = df[df["user_id"].isin(self.user_index) & df["app_id"].isin(self.game_index)]
        rows = df["user_id"].map(self.user_index).values
        cols = df["app_id"].map(self.game_index).values
        return csr_matrix((np.ones(len(df), dtype="float32"), (rows, cols)),
                          shape=(len(self.users), len(self.game_ids)))

    # ---------- Scoring ----------
    def _components(self, X):
        """Collaborative and content scores for the rows of X (users x games they liked)."""
        collaborative = (X @ self.collab_similarity).toarray().astype("float32")
        content = (X @ self.content_similarity).toarray().astype("float32")
        return collaborative, content

    def score(self, X, content_weight=None, popularity_weight=None):
        """
        Hybrid score = z(collaborative) + content_weight · z(content) − popularity_weight · z(popularity)
        Each part is standardized (z) so they are comparable before being added together.
        """
        wc = self.content_weight if content_weight is None else content_weight
        wp = self.popularity_weight if popularity_weight is None else popularity_weight
        collaborative, content = self._components(X)
        return _zscore(collaborative) + wc * _zscore(content) - wp * self.popularity_z

    def thematic_affinity(self, X, game_idx):
        """
        Cosine between each user's 'tag profile' and the tags of their games (0 = nothing, 1 = identical).
        X: users x games they liked. game_idx: users x k matrix with the game indices to compare.
        """
        profile = normalize(X @ self.tag_matrix)
        return np.take_along_axis((profile @ self.tag_matrix.T).toarray(), game_idx, axis=1)

    # ---------- Recommendation ----------
    def game_info(self, app_ids):
        cols = ["title", "year", "tags", "steam_rating", "price", "image"]
        info = self.games.reindex(app_ids)[cols].reset_index()
        info = info.dropna(subset=["title"])
        # If the game had no tags, show the estimated ones (flagged so they can be told apart)
        info["estimated_tags"] = (info["tags"] == "") & info["app_id"].isin(self.estimated_tags)
        info.loc[info["estimated_tags"], "tags"] = info.loc[info["estimated_tags"], "app_id"].map(self.estimated_tags)
        return info

    def is_known_user(self, user):
        return user in self.user_index and self.X_liked[self.user_index[user]].nnz > 0

    def _vector_from_games(self, app_ids):
        """1 x games row with a 1 in the given games (used for a hand-made 'user' profile)."""
        cols = [self.game_index[g] for g in app_ids if g in self.game_index]
        return csr_matrix((np.ones(len(cols), dtype="float32"), ([0] * len(cols), cols)),
                          shape=(1, len(self.game_ids)))

    def _recommend_from_vector(self, x, exclude, top_n, content_weight=None, popularity_weight=None):
        if x.nnz == 0:
            return self.global_top_info(top_n)

        scores = self.score(x, content_weight, popularity_weight)[0]
        exclude_idx = [self.game_index[g] for g in exclude if g in self.game_index]
        scores[exclude_idx] = -np.inf
        best = np.argsort(-scores)[:top_n]

        # Explanation: the liked game that is most similar to each recommendation
        # (empty if none of them is a neighbor, e.g. a game recommended mostly for its popularity)
        liked = x.indices
        similarity = (self.collab_similarity[liked][:, best] +
                      self.content_similarity[liked][:, best]).toarray()
        reasons = [self.games["title"].get(self.game_ids[liked[i]], "") if similarity[i, j] > 0 else ""
                   for j, i in enumerate(similarity.argmax(axis=0))]

        result = pd.DataFrame({"app_id": self.game_ids[best],
                               "affinity": self.thematic_affinity(x, best[np.newaxis])[0].round(3),
                               "because_you_liked": reasons})
        return self.game_info(result["app_id"].values).merge(result, on="app_id")

    def recommend(self, user, top_n=5, content_weight=None, popularity_weight=None):
        # If the user was not in the training data (e.g. filtered out), return the global top.
        if not self.is_known_user(user):
            return self.global_top_info(top_n)
        x = self.X_liked[self.user_index[user]]
        return self._recommend_from_vector(x, self.played.get(user, set()), top_n,
                                           content_weight, popularity_weight)

    def recommend_from_games(self, app_ids, top_n=5, content_weight=None, popularity_weight=None):
        """Recommendations for someone who tells us which games they like (does not need to be in the dataset)."""
        return self._recommend_from_vector(self._vector_from_games(app_ids), set(app_ids), top_n,
                                           content_weight, popularity_weight)

    def favorite_games(self, user, n=5):
        """Games the user recommended, sorted by hours played."""
        reviews = self.interactions[(self.interactions["user_id"] == user) & self.interactions["is_recommended"]]
        favorites = reviews.sort_values("hours", ascending=False).head(n)
        info = self.game_info(favorites["app_id"].values)
        return info.merge(favorites[["app_id", "hours"]], on="app_id")

    def similar_games(self, game, top_n=5, content_weight=None):
        """Games similar to a given one: a blend of collaborative and thematic similarity (both between 0 and 1)."""
        if game not in self.game_index:
            return pd.DataFrame()
        a = 0.5 if content_weight is None else content_weight
        j = self.game_index[game]
        collaborative = (self._games_by_users @ self._games_by_users[j].T).toarray().ravel()
        content = (self.tag_matrix @ self.tag_matrix[j].T).toarray().ravel()
        similarity = (1 - a) * collaborative + a * content
        similarity[j] = -np.inf
        best = np.argsort(-similarity)[:top_n]
        result = pd.DataFrame({"app_id": self.game_ids[best],
                               "similarity": similarity[best].round(3),
                               "affinity": content[best].round(3)})
        return self.game_info(result["app_id"].values).merge(result, on="app_id")

    def global_top_info(self, top_n=5):
        return self.game_info(self.global_top[:top_n])

    def find_game(self, text):
        """Returns the app_id of the model's game whose title contains the text (the most reviewed one)."""
        in_model = self.games.loc[self.games.index.intersection(self.game_ids)]
        matches = in_model[in_model["title"].str.contains(text, case=False, na=False, regex=False)]
        if matches.empty:
            return None
        return matches.sort_values("user_reviews", ascending=False).index[0]

    # ---------- (4) Evaluation with precision@k ----------
    def _evaluation_users(self, dataset, n_users):
        positives = dataset[dataset["is_recommended"] & dataset["user_id"].isin(self.user_index)
                            & dataset["app_id"].isin(self.game_index)]
        relevant = positives.groupby("user_id")["app_id"].apply(lambda s: {self.game_index[g] for g in s})
        relevant = relevant.sample(min(n_users, len(relevant)), random_state=SEED)
        return np.array([self.user_index[u] for u in relevant.index]), list(relevant.values)

    def evaluate(self, split="test", k=10, n_users=3000, models=None, weights=None):
        """
        For each sampled user: build their top-k (without the games they already reviewed in training)
        and count how many of the games they recommended in the evaluation set show up.
            precision@k  = hits / k
            recall@k     = hits / number of relevant games for the user
            coverage     = share of the catalog that appears in at least one top-k (variety)
            popularity   = median number of reviews of the recommended games (lower = less "the usual")
            affinity     = average thematic similarity between the recommendations and the user's taste
        """
        dataset = self.test if split == "test" else self.validation
        users, relevant = self._evaluation_users(dataset, n_users)

        exclude = self.X_seen
        if split == "test":  # In test, also skip what the user already saw in validation
            exclude = exclude + self._build_matrix(self.validation)

        models = models or ["Popularity", "SVD (collaborative)", "Item-item (collaborative)",
                            "Content (tags)", "Hybrid"]
        weights = weights or {}
        svd = None
        if "SVD (collaborative)" in models:
            svd_model = TruncatedSVD(n_components=SVD_FACTORS, random_state=SEED)
            user_factors = svd_model.fit_transform(self.X_liked)
            svd = (user_factors, svd_model.components_)

        batch_size = 500
        results = []
        for model in models:
            hits, recall, affinity, popularity, recommended = 0, 0.0, [], [], set()
            for start in range(0, len(users), batch_size):
                u = users[start:start + batch_size]
                X = self.X_liked[u]
                if model == "Popularity":
                    S = np.tile(self.popularity_z, (len(u), 1))
                elif model == "SVD (collaborative)":
                    S = svd[0][u] @ svd[1]
                elif model == "Item-item (collaborative)":
                    S = self.score(X, content_weight=0, popularity_weight=0)
                elif model == "Content (tags)":
                    S = _zscore(self._components(X)[1])
                else:
                    S = self.score(X, weights.get("content"), weights.get("popularity"))

                S = np.asarray(S, dtype="float32")
                no_data = np.asarray(X.sum(axis=1)).ravel() == 0  # No known taste -> popularity
                S[no_data] = self.popularity_z
                S[exclude[u].nonzero()] = -np.inf
                top = np.argpartition(-S, k, axis=1)[:, :k]

                affinity.extend(self.thematic_affinity(X, top).mean(axis=1))
                popularity.extend(np.median(self.popularity[top], axis=1))
                recommended.update(top.ravel())
                for games, rel in zip(top, relevant[start:start + batch_size]):
                    n_hits = len(set(games) & rel)
                    hits += n_hits
                    recall += n_hits / len(rel)

            n = len(users)
            results.append({"model": model,
                            f"precision@{k}": hits / (n * k),
                            f"recall@{k}": recall / n,
                            "coverage": len(recommended) / len(self.game_ids),
                            "median_popularity": float(np.median(popularity)),
                            "thematic_affinity": float(np.mean(affinity))})
        return pd.DataFrame(results)

    def tune_weights(self, k=10, n_users=2000,
                     content_weights=(0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.2), popularity_weights=(0, 0.15, 0.3, 0.5)):
        """Tries weight combinations on the VALIDATION set (never on test, to keep the test score honest)."""
        rows = []
        for wc in content_weights:
            for wp in popularity_weights:
                r = self.evaluate("validation", k, n_users, models=["Hybrid"],
                                  weights={"content": wc, "popularity": wp}).iloc[0]
                rows.append({"content_weight": wc, "popularity_weight": wp, **r.drop("model").to_dict()})
        return pd.DataFrame(rows)


def build_model():
    """Loads everything and returns the trained model together with the games and users tables."""
    required = ["games.csv", "users.csv"] + ([] if os.path.exists(CACHE_PATH) else ["recommendations.csv"])
    missing = [f for f in required if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        raise FileNotFoundError(f"Missing data files in {DATA_DIR}: {', '.join(missing)}. "
                                "Download them from Kaggle as explained in README.md.")

    games = load_games()
    users = load_users()
    interactions = load_interactions()
    model = HybridRecommender(interactions, games).fit()
    return model, games, users
