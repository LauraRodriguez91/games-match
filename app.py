import html
import pandas as pd
import streamlit as st
from recommender import build_model

# ---------------- Streamlit ----------------
# Web app to explore the recommendations interactively.
# Run with:  python -m streamlit run app.py
# Base widget colors live in .streamlit/config.toml; the card styles are added here.

st.set_page_config(page_title="GameMatch · Video Game Recommender", page_icon="🎮", layout="wide")

# ---------------- Styles (CSS) ----------------
st.markdown("""
<style>
:root {
    --purple: #a855f7;
    --light-purple: #c4b5fd;
    --green: #22c55e;
    --light-green: #4ade80;
    --background: #0f0b1a;
    --surface: #1a1433;
    --border: rgba(168, 85, 247, 0.25);
    --muted-text: #a59fbf;
}

.block-container { padding-top: 2rem; max-width: 1300px; }

/* Header */
.hero {
    background: linear-gradient(120deg, #3b0764 0%, #6d28d9 45%, #15803d 100%);
    border-radius: 20px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 40px rgba(109, 40, 217, 0.35);
}
.hero h1 { color: #fff; font-size: 2.4rem; margin: 0; font-weight: 800; letter-spacing: -0.5px; }
.hero p  { color: #e9e3ff; margin: .4rem 0 0 0; font-size: 1.05rem; }

/* Section titles */
.section {
    display: flex; align-items: center; gap: .6rem;
    font-size: 1.35rem; font-weight: 700; color: #f5f3ff;
    margin: 1.6rem 0 .8rem 0;
}
.section::before {
    content: ""; width: 6px; height: 1.4rem; border-radius: 3px;
    background: linear-gradient(180deg, var(--purple), var(--green));
}

/* Card grid */
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 1.1rem;
}
.card {
    position: relative;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    text-decoration: none !important;
    display: block;
}
.card:hover {
    transform: translateY(-4px);
    border-color: var(--green);
    box-shadow: 0 12px 30px rgba(34, 197, 94, 0.25);
}
.card img { width: 100%; aspect-ratio: 460 / 215; object-fit: cover; display: block; }
.card-body { padding: .8rem .9rem 1rem .9rem; }
.card-title { color: #fff; font-weight: 700; font-size: 1rem; line-height: 1.25; margin-bottom: .2rem; }
.meta { color: var(--muted-text); font-size: .8rem; margin-bottom: .55rem; }

.chips { display: flex; flex-wrap: wrap; gap: .3rem; margin-bottom: .6rem; }
.chip {
    background: rgba(168, 85, 247, 0.15); color: var(--light-purple);
    border: 1px solid rgba(168, 85, 247, 0.35);
    font-size: .7rem; padding: .1rem .5rem; border-radius: 999px;
}
/* Estimated tags (the game had no tags in the dataset) */
.chip.estimated { border-style: dashed; background: transparent; opacity: .85; }

.rank {
    position: absolute; top: .6rem; left: .6rem;
    background: linear-gradient(135deg, var(--purple), var(--green));
    color: #fff; font-weight: 800; font-size: .85rem;
    width: 2rem; height: 2rem; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 12px rgba(0,0,0,.4);
}

.score { display: flex; justify-content: space-between; font-size: .78rem; color: var(--muted-text); }
.score b { color: var(--light-green); }
.bar { height: 6px; background: rgba(255,255,255,.08); border-radius: 3px; margin-top: .3rem; overflow: hidden; }
.bar div { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--purple), var(--green)); }

.steam { font-size: .72rem; font-weight: 600; }
.steam.pos { color: var(--light-green); }
.steam.mix { color: #fbbf24; }
.steam.neg { color: #f87171; }

.reason { color: var(--muted-text); font-size: .75rem; margin-top: .55rem; line-height: 1.3; }
.reason b { color: var(--light-purple); font-weight: 600; }
.hours { color: var(--light-green); font-size: .8rem; font-weight: 600; }

.profile {
    display: flex; gap: 1rem; flex-wrap: wrap;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: .8rem 1.1rem; margin-top: .8rem;
    color: var(--muted-text); font-size: .9rem;
}
.profile b { color: #fff; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; }
.stTabs [data-baseweb="tab"] {
    background: var(--surface); border-radius: 10px 10px 0 0; padding: .5rem 1.1rem;
}
.stTabs [aria-selected="true"] { background: linear-gradient(120deg, #6d28d9, #15803d) !important; color: #fff !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(120deg, #7c3aed, #16a34a);
    color: #fff; border: none; border-radius: 10px; font-weight: 700; padding: .5rem 1.4rem;
}
.stButton > button:hover { filter: brightness(1.15); color: #fff; }

/* No sidebar */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading data and training the model...")
def get_model():
    # Runs only once: Streamlit keeps the model in memory between interactions.
    model, games, users = build_model()

    # Games in the model, most popular first, as {app_id: label} for the dropdowns.
    # Some titles are repeated in Steam, so the year is added to tell them apart.
    in_model = games[games["app_id"].isin(model.game_ids)].sort_values("user_reviews", ascending=False)
    repeated = in_model["title"].duplicated(keep=False)
    labels = in_model["title"].where(~repeated, in_model["title"] + " (" + in_model["year"].astype(str) + ")")
    return model, users, dict(zip(in_model["app_id"], labels))


model, users, game_labels = get_model()


# ---------------- UI components ----------------

def section_title(text):
    st.markdown(f'<div class="section">{html.escape(text)}</div>', unsafe_allow_html=True)


def steam_rating_class(rating):
    r = str(rating).lower()
    if "positive" in r:
        return "pos"
    if "mixed" in r:
        return "mix"
    return "neg"


def game_card(row, rank=None, score=None):
    """Returns the HTML of a game card. score = (label, value, maximum)."""
    title = html.escape(str(row["title"]))
    price = "Free" if row["price"] == 0 else f"${row['price']:.2f}"
    rating = (f'<span class="steam {steam_rating_class(row["steam_rating"])}">'
              f'{html.escape(str(row["steam_rating"]))}</span>')
    meta = " · ".join(([str(int(row["year"]))] if pd.notna(row["year"]) else []) + [price, rating])
    estimated = bool(row.get("estimated_tags", False))
    chip_class = "chip estimated" if estimated else "chip"
    tooltip = ' title="Estimated tag: this game had no tags in the dataset"' if estimated else ""
    chips = "".join(f'<span class="{chip_class}"{tooltip}>{html.escape(t.strip())}</span>'
                    for t in str(row["tags"]).split(",") if t.strip())

    parts = [f'<a class="card" href="https://store.steampowered.com/app/{row["app_id"]}" '
             f'target="_blank" rel="noopener noreferrer">']
    if rank is not None:
        parts.append(f'<div class="rank">{rank}</div>')
    parts.append(f'<img src="{row["image"]}" alt="{title}" loading="lazy">')
    parts.append('<div class="card-body">')
    parts.append(f'<div class="card-title">{title}</div>')
    parts.append(f'<div class="meta">{meta}</div>')
    if chips:
        parts.append(f'<div class="chips">{chips}</div>')
    if pd.notna(row.get("hours")):
        parts.append(f'<div class="hours">⏱ {row["hours"]:.0f} h played</div>')
    if score is not None:
        label, value, maximum = score
        percentage = max(0, min(100, value / maximum * 100))
        value_text = f"{value:.0%}" if maximum == 1 else f"{value:.1f}/{maximum}"
        parts.append(f'<div class="score"><span>{label}</span><b>{value_text}</b></div>')
        parts.append(f'<div class="bar"><div style="width:{percentage:.0f}%"></div></div>')
    if isinstance(row.get("because_you_liked"), str) and row["because_you_liked"]:
        parts.append(f'<div class="reason">Because you liked <b>{html.escape(row["because_you_liked"])}</b></div>')
    parts.append('</div></a>')
    return "".join(parts)


def show_games(df, column=None, label=None, maximum=10, numbered=False):
    """Shows the games as a grid of cards with their Steam header image."""
    if df.empty:
        st.info("No games to show.")
        return
    cards = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        score = (label, float(row[column]), maximum) if column else None
        cards.append(game_card(row, rank=i if numbered else None, score=score))
    st.markdown(f'<div class="grid">{"".join(cards)}</div>', unsafe_allow_html=True)


# ---------------- Header ----------------
st.markdown("""
<div class="hero">
    <h1>GameMatch</h1>
    <p>You're just a click away from your new favorite game.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Navigation ----------------
# st.tabs does not remember the open tab across reruns, so it jumps back to the first one.
# A segmented control with a key keeps the selected section in st.session_state.
SECTIONS = ["👤  For a user", "🧩  Build your profile", "🔍  Similar games", "🏆  Global top"]
section = st.segmented_control("Section", SECTIONS, default=SECTIONS[0], key="section",
                               label_visibility="collapsed")
if section is None:  # Clicking the selected option again deselects it
    section = SECTIONS[0]

# ---------------- Recommendations for a user ----------------
if section == SECTIONS[0]:
    col_mode, col_user = st.columns([1, 2])
    with col_mode:
        mode = st.radio("Choose a user", ["From the list", "Type an ID"], horizontal=True)
    with col_user:
        if mode == "From the list":
            # Only users with known tastes, and only a sample: there are too many for a dropdown.
            selected_user = st.selectbox("User", model.known_users[:5000].tolist())
        else:
            selected_user = st.number_input("User ID", min_value=0, step=1, value=0)

    # Store the user in the session so the results stay visible when other widgets are used
    if st.button("✨ Get recommendations"):
        st.session_state["active_user"] = selected_user

    if st.session_state.get("active_user") is not None:
        selected_user = st.session_state["active_user"]
        info = users[users["user_id"] == selected_user]
        if not info.empty:
            st.markdown(
                f'<div class="profile">👤 User <b>{selected_user}</b>'
                f'<span>🎮 <b>{int(info["library_size"].iloc[0])}</b> games in library</span>'
                f'<span>📝 <b>{int(info["review_count"].iloc[0])}</b> reviews written</span></div>',
                unsafe_allow_html=True)

        if model.is_known_user(selected_user):
            section_title("Their favorite games")
            show_games(model.favorite_games(selected_user))

            section_title("Recommended for them")
            show_games(model.recommend(selected_user, top_n=10),
                       column="affinity", label="Thematic affinity", maximum=1, numbered=True)
        else:
            st.warning("We don't have enough data about this user's taste to personalize the recommendations. "
                       "Here is our global top 5:")
            show_games(model.global_top_info(5), numbered=True)

# ---------------- Hand-made profile ----------------
elif section == SECTIONS[1]:
    chosen = st.multiselect("Pick a few games you like and we'll recommend others:",
                            list(game_labels), format_func=game_labels.get, max_selections=20,
                            key="profile_games", placeholder="Type to search... (e.g. Hollow Knight, Stardew Valley)")
    if chosen:
        section_title("For you")
        show_games(model.recommend_from_games(chosen, top_n=10),
                   column="affinity", label="Thematic affinity", maximum=1, numbered=True)

# ---------------- Similar games ----------------
elif section == SECTIONS[2]:
    app_id = st.selectbox("Pick a game you like", list(game_labels), format_func=game_labels.get,
                          key="similar_game")
    if app_id is not None:
        col_image, col_text = st.columns([1, 2])
        with col_image:
            show_games(model.game_info([app_id]))
        with col_text:
            section_title(f"If you like {game_labels[app_id]}, try:")
            blend = st.slider("Players in common  ↔  Similar theme", 0.0, 1.0, 0.5, 0.1, key="similar_blend")
        st.markdown('<div style="height: 60px;"></div>', unsafe_allow_html=True)  # Space before the results
        show_games(model.similar_games(app_id, top_n=10, content_weight=blend),
                   column="similarity", label="Similarity", maximum=1, numbered=True)

# ---------------- Global top ----------------
else:
    section_title("Best rated among the most popular")
    show_games(model.global_top_info(10), numbered=True)