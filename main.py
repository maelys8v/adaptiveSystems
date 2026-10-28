"""Adaptive Systems - simple version of the project.

This file addresses tasks 1 through 4 of the assignment:

1. TF-IDF on descriptions + ratio_quality calculation for "Food & Drink".
2. The same with LDA (30 topics, 2 passes, random_state=30).
3. The same for "Sports" and comparison with "Food & Drink".
4. Article similarity based on tags (Jaccard and TF-IDF + cosine).

Task 5 is a design question (content-based RS): it requires a written
answer and is therefore not implemented here.

Important implementation choices:

- Use Gensim for Dictionary, TF-IDF, LDA, and MatrixSimilarity.
- Use NLTK for tokenization and stemming.
- Deliberately keep a single file to make the project easier to explain.
- Follow the assignment pseudocode literally, so an article may appear
  among its own 10 nearest neighbors (similarity 1.0).
  This is what the pseudocode does. For a more realistic measure,
  simply set `exclude_self=True` in `ratio_quality`.
"""

import time

import numpy as np
import pandas as pd
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
CSV_PATH = "news2.csv"
FOOD_TOPIC = "Food & Drink"
SPORTS_TOPIC = "Sports"
TOP_K = 10

# Stemming reduces the vocabulary: "recipes", "recipe", and "reciped" become the same token.
# This makes TF-IDF vectors more stable.
STEMMER = PorterStemmer()


def preprocess(text):
    """Convert a description into a list of normalized tokens.

    Steps:
    1. lowercase;
    2. tokenize with NLTK (wordpunct_tokenize requires no NLTK data
       downloads, unlike word_tokenize);
    3. remove punctuation and numbers;
    4. remove tokens that are too short;
    5. remove stop words (very frequent, uninformative words);
    6. apply stemming with NLTK.

    Use Gensim's stop word list to avoid downloading NLTK's `stopwords`
    corpus. NLTK is still used for tokenization and stemming, as required
    by the assignment.
    """

    tokens = wordpunct_tokenize(str(text).lower())

    cleaned = []
    for token in tokens:
        # Keep only sequences of letters (no "123", "!", etc.).
        if not token.isalpha():
            continue

        # A single-letter token is almost always noise.
        if len(token) < 2:
            continue

        # Stop words appear everywhere and do not help distinguish topics.
        if token in STOPWORDS:
            continue

        cleaned.append(STEMMER.stem(token))

    return cleaned


def load_data():
    """Load the CSV and remove unusable rows.

    Remove empty descriptions because they cannot produce a vector.
    Reset the index so list positions match positions in the DataFrame.
    """

    df = pd.read_csv(CSV_PATH)

    df = df.dropna(subset=["description", "article_section"]).copy()
    df["description"] = df["description"].astype(str)
    df["article_section"] = df["article_section"].astype(str).str.strip()

    df = df[df["description"].str.strip() != ""]
    df = df.reset_index(drop=True)

    return df


def build_tfidf(descriptions):
    """Build the TF-IDF model and similarity index.

    Returns:
    - `vectors`: the TF-IDF vectors for each article;
    - `similarity`: the MatrixSimilarity object;
    - `elapsed`: model creation time in seconds.

    Timing starts before preprocessing and stops immediately after
    `similarities.MatrixSimilarity(...)`, exactly as required by the
    assignment.
    """

    start = time.perf_counter()

    # 1. Preprocess all descriptions.
    texts = [preprocess(description) for description in descriptions]

    # 2. Dictionary: map each token to an integer identifier.
    dictionary = corpora.Dictionary(texts)

    # Remove words that appear in fewer than 2 documents (noise)
    # and those appearing in more than 90% of documents (too general).
    dictionary.filter_extremes(no_below=2, no_above=0.9)

    # 3. Bag-of-Words: each document becomes a list of (token_id, frequency).
    bow_corpus = [dictionary.doc2bow(tokens) for tokens in texts]

    # 4. TF-IDF: rare, discriminative words receive more weight.
    # `normalize=True` gives vectors a norm of 1, which is the
    # usual configuration for cosine similarity.
    tfidf_model = models.TfidfModel(bow_corpus, normalize=True)
    tfidf_vectors = list(tfidf_model[bow_corpus])

    # 5. Similarity index: this is the explicitly required call.
    similarity = similarities.MatrixSimilarity(tfidf_vectors)

    elapsed = time.perf_counter() - start
    return tfidf_vectors, similarity, elapsed


def build_lda(descriptions):
    """Build the LDA model and similarity index.

    Parameters required by the assignment:
    - 30 topics;
    - 2 passes;
    - random_state=30.

    Filter the dictionary slightly more than for TF-IDF: LDA is a
    probabilistic model, and very rare words make topics noisier.
    """

    # As with TF-IDF, timing includes preprocessing for this model.
    # This stays close to the assignment definition: "from the start of
    # the program up to the call to MatrixSimilarity".
    start = time.perf_counter()

    texts = [preprocess(description) for description in descriptions]

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    bow_corpus = [dictionary.doc2bow(tokens) for tokens in texts]

    # Train LDA.
    lda_model = models.LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=30,
        passes=2,
        random_state=30,
    )

    # Each document is transformed into a distribution over the 30 topics.
    # `minimum_probability=0.0` matters: otherwise Gensim removes small
    # probabilities and the vectors no longer all have the same dimension.
    lda_vectors = [
        lda_model.get_document_topics(bow, minimum_probability=0.0)
        for bow in bow_corpus
    ]

    similarity = similarities.MatrixSimilarity(lda_vectors)

    elapsed = time.perf_counter() - start
    return lda_vectors, similarity, elapsed


def ratio_quality(similarity_for_query, labels, topic, top_k=TOP_K,
                  exclude_self=False):
    """Implement the assignment pseudocode.

    `similarity_for_query(i)` must return an array of scores between
    article `i` and all articles in the corpus.

    Returns (ratio_quality, pseudocode_execution_time, total_goods).

    `exclude_self=False` by default: follow the pseudocode literally.
    The article is in the corpus, so it is its own most similar neighbor.
    This slightly inflates the ratio (at least 1 good neighbor out of 10).
    Set `exclude_self=True` for a more realistic measure.
    """

    # Articles in the requested section serve as queries.
    query_indices = [
        index
        for index, label in enumerate(labels)
        if label == topic
    ]

    # If the section does not exist (for example in a test subset),
    # return zero instead of dividing by zero.
    if not query_indices:
        return 0.0, 0.0, 0

    total_goods = 0

    # Time only the execution of the pseudocode, excluding model
    # creation.
    start = time.perf_counter()

    for query_index in query_indices:
        # Similarity scores between the query and all articles.
        scores = np.asarray(similarity_for_query(query_index))

        # argsort sorts in ascending order; the minus sign gives descending
        # order (most similar first).
        ranked = np.argsort(-scores)

        # Optionally remove the query itself from its neighbors.
        if exclude_self:
            ranked = ranked[ranked != query_index]

        # The top_k neighbors.
        top_indices = ranked[:top_k]

        # Count how many of these neighbors belong to the same section.
        goods = sum(
            1 for index in top_indices
            if labels[int(index)] == topic
        )
        total_goods += goods

    elapsed = time.perf_counter() - start

    # Exact formula from the pseudocode.
    quality = total_goods / (len(query_indices) * top_k)

    return quality, elapsed, total_goods


def parse_tags(raw_tags):
    """Convert "Taste ,rice, dessert" into {"taste", "rice", "dessert"}.

    Lowercase and strip whitespace so tags are comparable even if the
    CSV contains inconsistent spacing.
    """

    # If the field is empty/NaN, there are no tags to compare.
    if pd.isna(raw_tags):
        return set()

    tags = set()
    for raw_tag in str(raw_tags).split(","):
        tag = raw_tag.strip().lower()
        if tag:
            tags.add(tag)
    return tags


def jaccard(tags_a, tags_b):
    """Jaccard similarity between two sets of tags.

    J(A, B) = |A intersection B| / |A union B|

    This is a natural method for tags: it equals 1 when both articles
    have exactly the same tags, and 0 when they share none.
    """

    union = tags_a | tags_b
    if not union:
        return 0.0
    return len(tags_a & tags_b) / len(union)


def tag_scores(query_index, tag_sets):
    """Compute Jaccard similarity between an article and all other articles."""

    query_tags = tag_sets[query_index]
    return np.array([
        jaccard(query_tags, other_tags)
        for other_tags in tag_sets
    ])


def build_tag_tfidf(raw_tags):
    """Build the TF-IDF model on tags and the similarity index.

    Returns:
    - `vectors`: the tag TF-IDF vectors for each article;
    - `similarity`: the MatrixSimilarity object for cosine similarity;
    - `elapsed`: preparation and indexing time in seconds.

    Each complete tag is a vocabulary unit, even if it contains several
    words. Only tags are used to build the model.

    Timing includes tag normalization, dictionary creation, TF-IDF
    calculation, and index construction.
    """

    start = time.perf_counter()

    # 1. Normalize tags using the same function as for Jaccard.
    # A complete tag remains one token: "climate change" is not split.
    # Sorting gives the tags in each set a stable order.
    texts = [sorted(parse_tags(tags)) for tags in raw_tags]

    # 2. Dictionary: map each complete tag to an integer identifier.
    # Keep all tags; IDF reduces the weight of the most frequent ones.
    dictionary = corpora.Dictionary(texts)

    # 3. Bag-of-Words: each article becomes a list of (tag_id, frequency).
    # parse_tags returns a set: each tag appears at most once.
    bow_corpus = [dictionary.doc2bow(tags) for tags in texts]

    # 4. TF-IDF: tags that are rare in the corpus receive more weight.
    # normalize=True gives nonzero vectors a norm of 1.
    tfidf_model = models.TfidfModel(bow_corpus, normalize=True)
    tfidf_vectors = list(tfidf_model[bow_corpus])

    # 5. Cosine similarity index for the articles' TF-IDF vectors.
    # num_features fixes the dimension even if some IDF weights are zero.
    # If no tags exist, the index contains only zero vectors.
    # max(1, ...) then provides the minimum dimension required by Gensim.
    similarity = similarities.MatrixSimilarity(
        tfidf_vectors, num_features=max(1, len(dictionary))
    )

    elapsed = time.perf_counter() - start
    return tfidf_vectors, similarity, elapsed


def print_topic_results(model_name, labels, similarity_for_query,
                        model_creation_time):
    """Display results for the two requested sections."""

    print(f"\n--- {model_name} ---")
    print(f"Model creation time: {model_creation_time:.2f} s")

    for topic in (FOOD_TOPIC, SPORTS_TOPIC):
        quality, pseudo_time, total_goods = ratio_quality(
            similarity_for_query, labels, topic
        )
        print(
            f"  {topic} : ratio_quality = {quality:.4f} "
            f"(total_goods = {total_goods}, "
            f"pseudocode = {pseudo_time:.2f} s)"
        )


def main():
    """Program entry point."""

    print("Loading data...")
    df = load_data()
    descriptions = df["description"].tolist()
    labels = df["article_section"].tolist()

    print(f"{len(df)} usable articles")
    print(f"{labels.count(FOOD_TOPIC)} articles 'Food & Drink'")
    print(f"{labels.count(SPORTS_TOPIC)} articles 'Sports'")

    # ------------------------------------------------------------------
    # Task 1: TF-IDF
    # ------------------------------------------------------------------
    print("\n=== Task 1: TF-IDF ===")
    tfidf_vectors, tfidf_similarity, tfidf_time = build_tfidf(descriptions)

    # Pass a small function to ratio_quality so it can retrieve the scores
    # for a given article. This avoids duplicating the pseudocode for
    # TF-IDF, LDA, and tags.
    print_topic_results(
        "TF-IDF",
        labels,
        lambda index: tfidf_similarity[tfidf_vectors[index]],
        tfidf_time,
    )

    # ------------------------------------------------------------------
    # Task 2: LDA
    # ------------------------------------------------------------------
    print("\n=== Task 2: LDA (30 topics, 2 passes, random_state=30) ===")
    lda_vectors, lda_similarity, lda_time = build_lda(descriptions)

    print_topic_results(
        "LDA",
        labels,
        lambda index: lda_similarity[lda_vectors[index]],
        lda_time,
    )

    # ------------------------------------------------------------------
    # Task 3: Sports vs Food & Drink
    # ------------------------------------------------------------------
    # The "Sports" results are already displayed in the two preceding
    # sections. Compare the two lines to answer question 3:
    # the Sports ratio is lower because sports vocabulary often overlaps
    # with politics, entertainment, or celebrities.
    print("\n=== Task 3 ===")
    print("Compare the 'Sports' and 'Food & Drink' lines above.")

    # ------------------------------------------------------------------
    # Task 4: tag similarity (Jaccard and TF-IDF + cosine)
    # ------------------------------------------------------------------
    print("\n=== Task 4: Jaccard vs TF-IDF on tags ===")

    # Use the same corpus for both methods, retaining articles without tags.
    start = time.perf_counter()
    tag_sets = [parse_tags(tags) for tags in df["tags"]]
    jaccard_time = time.perf_counter() - start

    tag_vectors, tag_similarity, tag_tfidf_time = build_tag_tfidf(df["tags"])
    print(f"Jaccard preparation time: {jaccard_time:.2f} s")
    print(f"Tag TF-IDF preparation/indexing time: {tag_tfidf_time:.2f} s")

    for topic in (FOOD_TOPIC, SPORTS_TOPIC):
        # Identical settings, following the assignment pseudocode.
        jaccard_quality, jaccard_eval_time, _ = ratio_quality(
            lambda index: tag_scores(index, tag_sets),
            labels, topic, top_k=TOP_K, exclude_self=False,
        )
        tfidf_quality, tfidf_eval_time, _ = ratio_quality(
            lambda index: tag_similarity[tag_vectors[index]],
            labels, topic, top_k=TOP_K, exclude_self=False,
        )
        print(f"\n  {topic} (top-{TOP_K}) :")
        print(
            f"    Jaccard : ratio_quality = {jaccard_quality:.4f} "
            f"(calculation = {jaccard_eval_time:.2f} s)"
        )
        print(
            f"    Tag TF-IDF + cosine: ratio_quality = {tfidf_quality:.4f} "
            f"(calculation = {tfidf_eval_time:.2f} s)"
        )
        print(
            f"    Difference (Tag TF-IDF - Jaccard): "
            f"{tfidf_quality - jaccard_quality:+.4f}"
        )

    print(
        "\nNote: TF-IDF gives more weight to shared rare tags "
        "than to very frequent tags. Any improvement depends on the measured "
        "results; two articles with no shared tags still have zero similarity."
    )


if __name__ == "__main__":
    main()
