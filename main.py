from tabulate import tabulate
from recommender import build_model, CACHE_PATH, CONTENT_WEIGHT, POPULARITY_WEIGHT

TUNE_WEIGHTS = False  # Set to True to rerun the weight search on the validation set (takes a few minutes)


def show_table(df):
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))


# ---------------- Data loading and training ----------------
# The first run reads the full recommendations.csv (~2 GB) and can take several minutes.
# A filtered cache is then saved in data/ and later runs are much faster.

model, games, users = build_model()
data = model.interactions

print(f"Reviews used: {len(data):,} | Users: {data['user_id'].nunique():,} | "
      f"Games: {data['app_id'].nunique():,} | Positive: {data['is_recommended'].mean():.0%}")
print(f"Games without tags in games_metadata.json (tags estimated from their neighbors): {model.n_estimated_tags}")
print(f"Hybrid model weights -> content: {CONTENT_WEIGHT} | popularity penalty: {POPULARITY_WEIGHT}")
print(f"(Filtered reviews cache: {CACHE_PATH})\n")

# ---------------- Weight tuning (validation set) ----------------
if TUNE_WEIGHTS:
    print("Weight tuning on VALIDATION:")
    show_table(model.tune_weights().round(4))

# ---------------- Evaluation (test set) ----------------
# The hybrid model is compared with each of its parts on their own and with two baselines:
# recommending the most popular games and a TruncatedSVD model.
print("Evaluation on TEST (3,000 users, top-10):")
show_table(model.evaluate("test", k=10, n_users=3000).round(4))

columns = ["title", "year", "tags"]

# ---------------- Example with a user known by the model ----------------
EXAMPLE_USER = 752555  # A user with varied, easy-to-read tastes
known_user = EXAMPLE_USER if model.is_known_user(EXAMPLE_USER) else model.known_users[0]
user_info = users[users["user_id"] == known_user]

print("\nKnown user:", known_user)
if not user_info.empty:
    print("Games in library:", int(user_info["library_size"].iloc[0]),
          "| Reviews:", int(user_info["review_count"].iloc[0]))

# The user's favorite games, to compare with the recommendations:
print("User's favorite games (recommended by them, sorted by hours played):")
show_table(model.favorite_games(known_user, n=5)[columns + ["hours"]])

print("Top-5 recommendations:")
show_table(model.recommend(known_user)[columns + ["affinity", "because_you_liked"]])

# ---------------- Example with an unknown user ----------------
new_user = -1  # An ID that does not appear in the training data
print("\nUser unknown to the model:", new_user)
print("Global top-5:")
show_table(model.recommend(new_user)[columns])

# ---------------- Hand-made profile ----------------
# Someone who is not in the dataset tells us which games they like:
my_games = [model.find_game(t) for t in ["Resident Evil Village", "Phasmophobia", "Outlast"]]
my_games = [g for g in my_games if g is not None]
print("\nHand-made profile with:", ", ".join(games.loc[games["app_id"].isin(my_games), "title"]))
show_table(model.recommend_from_games(my_games, top_n=5)[columns + ["affinity", "because_you_liked"]])

# ---------------- Similar games ----------------
game = model.find_game("Portal 2")
if game is None:
    game = model.global_top[0]
print("\nGames similar to:", games.loc[games["app_id"] == game, "title"].iloc[0])
show_table(model.similar_games(game)[columns + ["similarity", "affinity"]])
