"""Task 5: a local, four-screen Streamlit news recommender.

Run with: python -m streamlit run prototype.py
The experiment in main.py is protected by its __main__ guard, so importing
its tag helpers does not run Tasks 1-4.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from main import parse_tags, tag_scores


CSV_PATH = Path(__file__).with_name("news2.csv")
# Read-only scenario shared by the four pages. These are original CSV row IDs.
SCENARIO_SECTIONS = ["Sports", "Tech", "Food & Drink"]
SCENARIO_ARTICLES = {"Sports": 933, "Tech": 2702, "Food & Drink": 288,
                     "Entertainment": 19, "Politics": 89}
EXAMPLE_LIKES = [933, 288]
EXAMPLE_DISLIKES = [19, 89]
EXAMPLE_READ = [933, 288, 2702, 19]


@st.cache_resource(show_spinner="Loading articles and preparing tags...")
def load_catalogue(csv_path, modified_time):
    """Cache shared article data/tags, never personal feedback.

    modified_time is a cache key: changing the CSV reloads the tags.
    IDs are original CSV row positions, kept even when invalid rows are removed.
    Each tag remains a complete label, normalized by main.parse_tags().
    """
    articles = pd.read_csv(csv_path).fillna("")
    required = {"headline", "description", "article_section", "tags"}
    if not required.issubset(articles.columns):
        raise ValueError("news2.csv must contain: " + ", ".join(sorted(required)))
    articles["article_id"] = np.arange(len(articles))
    for column in required:
        articles[column] = articles[column].astype(str).str.strip()
    articles = articles[
        articles["description"].ne("") & articles["article_section"].ne("")
    ].reset_index(drop=True)
    if articles.empty:
        raise ValueError("news2.csv contains no usable descriptions and sections.")
    # Immutable tag sets prevent accidental changes to the shared catalogue.
    tag_sets = [frozenset(parse_tags(raw_tags)) for raw_tags in articles["tags"]]
    return articles, tag_sets





def go_to(screen):
    st.session_state.screen = screen


def open_article(article_id):
    """Navigate only: do not record an opening or update the feed."""
    st.session_state.current_article = article_id
    st.session_state.screen = "Article details"





def feedback_controls(article_id, prefix, columns=None):
    """Predefined visual states illustrate Alex's fictional history only."""
    left, right = columns if columns is not None else st.columns(2)
    liked = article_id in EXAMPLE_LIKES
    disliked = article_id in EXAMPLE_DISLIKES
    left.button("Like", key=f"{prefix}_like_{article_id}",
                type="primary" if liked else "secondary", disabled=True)
    right.button("Not Interested", key=f"{prefix}_dislike_{article_id}",
                 type="primary" if disliked else "secondary", disabled=True)
    if prefix == "details" and (liked or disliked):
        st.caption("Predefined example feedback: " + ("Liked" if liked else "Not Interested"))


def show_article(row, prefix, explanation=None):
    """Text-only feed card using only the supplied fields."""
    article_id = int(row.article_id)
    with st.container(border=True):
        st.subheader(row.headline or "(No headline supplied)")
        st.caption(f"Section: {row.article_section}")
        length = 400
        snippet = row.description[:length]
        st.text("Description: " + snippet + ("…" if len(row.description) > length else ""))
        tags = row.tags
        st.text("Tags: " + (tags or "None supplied"))
        if explanation:
            st.caption(explanation)
        open_column, like_column, dislike_column = st.columns([1.3, 1, 1.2])
        open_column.button("Open description", key=f"{prefix}_open_{article_id}",
                           on_click=open_article, args=(article_id,))
        feedback_controls(article_id, prefix, (like_column, dislike_column))


def onboarding(sections):
    st.header("1 - Initial preferences")
    st.write("Alex's initial interests: Sports, Tech and Food & Drink.")
    st.multiselect("Preferred sections", sections, default=SCENARIO_SECTIONS, disabled=True)
    st.caption("Fixed example: in the proposed system, readers would choose their sections.")
    st.button("Start reading", on_click=go_to, args=("Recommended feed",))


def feed(articles):
    st.header("2 - Recommended feed")
    st.write("Three predefined articles illustrating Alex's news feed.")
    for section in SCENARIO_SECTIONS:
        rows = articles.loc[articles["article_id"].eq(SCENARIO_ARTICLES[section])]
        if not rows.empty:
            show_article(rows.iloc[0], "feed", f"Because Alex selected {section}.")


def details(articles, tag_sets):
    st.header("3 · Article details")
    article_id = st.session_state.current_article
    position = articles.index[articles["article_id"].eq(article_id)][0]
    row = articles.iloc[position]
    st.subheader(row.headline or "(No headline supplied)")
    st.caption(f"Section: {row.article_section}")
    st.text("Tags: " + (row.tags or "None supplied"))
    st.markdown("**Description**")
    st.text(row.description)
    st.caption("This is the supplied description. Full article text is not available in this dataset.")
    feedback_controls(article_id, "details")
    st.button("Back to feed", on_click=go_to, args=("Recommended feed",))

    st.subheader("Related articles")
    st.caption("Jaccard similarity between article tags; only shared-tag matches are shown.")
    scores = tag_scores(position, tag_sets)
    related = [int(i) for i in np.argsort(-scores, kind="stable")
               if int(i) != position and scores[i] > 0][:2]
    if not related:
        st.info("No related articles with shared tags.")
    for index in related:
        row = articles.iloc[index]
        # Links change the viewed article only, not the scenario data.
        st.button(row.headline or "(No headline supplied)", key=f"related_open_{row.article_id}",
                  on_click=open_article, args=(int(row.article_id),))


def example_history(articles):
    """Fixed fictional interactions, attached to real CSV articles for display.

    These examples are never stored as feedback or used to rank the feed.
    """
    def rows(ids):
        indexed = articles.set_index("article_id", drop=False)
        return indexed.loc[[i for i in ids if i in indexed.index]]

    return {"Liked": rows(EXAMPLE_LIKES), "Not Interested": rows(EXAMPLE_DISLIKES),
            "Opened (read history)": rows(EXAMPLE_READ)}


def interests(articles, sections):
    st.header("4 - My interests")
    st.multiselect("Preferred sections", sections, default=SCENARIO_SECTIONS, disabled=True)
    st.button("Save preferences", disabled=True)
    st.write("Selected sections: " + ", ".join(SCENARIO_SECTIONS))

    st.subheader("Alex · example profile")
    st.caption("Fixed fictional history · real articles · no feedback learning")
    history = example_history(articles)
    columns = st.columns(3)
    for column, (name, items) in zip(columns, history.items()):
        column.metric(name, len(items))

    st.markdown("**Learned interests (illustrative)**")
    liked = history["Liked"]
    st.write("Example sections: " + ", ".join(liked["article_section"].unique()))
    example_tags = set()
    for raw_tags in liked["tags"]:
        example_tags.update(parse_tags(raw_tags))
    st.text("Example tags: " + (", ".join(sorted(example_tags)[:8]) or "None supplied"))
    st.caption("Sports and Food & Drink reflect the example likes; Tech is an initial preference.")

    st.subheader("Article history (fictional example)")
    for name, items in history.items():
        with st.expander(f"{name} ({len(items)})", expanded=True):
            # Read-only tables illustrate the stored profile information.
            st.dataframe(items[["headline", "article_section"]].reset_index(drop=True),
                         hide_index=True)
    sample_id = SCENARIO_ARTICLES["Sports"]
    st.button("Revisit example liked article", on_click=open_article, args=(sample_id,))
    st.caption("This example profile is fixed and cannot be edited.")


def main():
    st.set_page_config(page_title="Task 5 · News recommender", layout="centered")
    st.title("News recommender")
    st.caption("Illustrative prototype · real articles · fictional user interactions")
    st.caption("Read-only demonstration: use the navigation to explore the four pages.")
    if "screen" not in st.session_state:
        st.session_state.screen = "Initial preferences"
        st.session_state.current_article = SCENARIO_ARTICLES["Sports"]
    # Remove editable state left by previous app versions.
    for key in ("profile", "report_view", "feed_page"):
        st.session_state.pop(key, None)
    try:
        articles, tag_sets = load_catalogue(str(CSV_PATH), CSV_PATH.stat().st_mtime_ns)
    except (OSError, ValueError) as error:
        st.error(f"Cannot load the article catalogue: {error}")
        st.stop()
    sections = sorted(articles["article_section"].unique())
    if st.session_state.get("current_article") is None:
        st.session_state.current_article = SCENARIO_ARTICLES["Sports"]
    with st.sidebar:
        st.write("Alex · example reader")
        st.button("Initial preferences", on_click=go_to, args=("Initial preferences",))
        st.button("Recommended feed", on_click=go_to, args=("Recommended feed",))
        st.button("Article details", on_click=open_article, args=(SCENARIO_ARTICLES["Sports"],))
        st.button("My interests", on_click=go_to, args=("My interests",))
        st.caption("Preferences, feedback and history are predefined and read-only.")
    screen = st.session_state.screen
    if screen == "Initial preferences":
        onboarding(sections)
    elif screen == "Recommended feed":
        feed(articles)
    elif screen == "Article details":
        details(articles, tag_sets)
    else:
        interests(articles, sections)
    st.divider()
    st.caption("Descriptions are provided by the dataset. No publication dates are available.")


if __name__ == "__main__":
    main()
