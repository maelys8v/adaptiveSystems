"""
Adaptive Systems - version simple du projet.

Ce fichier répond aux tâches 1 à 4 de l'énoncé :

1. TF-IDF sur les descriptions + calcul de ratio_quality pour "Food & Drink".
2. Même chose avec LDA (30 topics, 2 passes, random_state=30).
3. Même chose pour "Sports" et comparaison avec "Food & Drink".
4. Similarité entre articles à partir des tags (Jaccard).

La tâche 5 est une question de conception (RS content-based) : elle est
rédactionnelle et n'est donc pas codée ici.

Choix d'implémentation importants :

- On utilise Gensim pour Dictionary, TF-IDF, LDA et MatrixSimilarity.
- On utilise NLTK pour la tokenisation et le stemming.
- On garde volontairement un seul fichier : c'est plus simple à expliquer.
- On suit le pseudo-code de l'énoncé littéralement, donc l'article lui-même
  peut apparaître dans ses propres 10 plus proches voisins (similarité 1.0).
  C'est ce que fait le pseudo-code. Si on veut une mesure plus
  honnête, il suffit de mettre `exclude_self=True` dans `ratio_quality`.
"""

import time

import numpy as np
import pandas as pd
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

# ---------------------------------------------------------------------------
# Paramètres de l'expérience
# ---------------------------------------------------------------------------
CSV_PATH = "news2.csv"
FOOD_TOPIC = "Food & Drink"
SPORTS_TOPIC = "Sports"
TOP_K = 10

# Le stemming réduit le vocabulaire : "recipes", "recipe" et "reciped" deviennent le même token.
# Cela rend les vecteurs TF-IDF plus stables.
STEMMER = PorterStemmer()


def preprocess(text):
    """Transforme une description en liste de tokens normalisés.

    Étapes :
    1. minuscules ;
    2. tokenisation avec NLTK (wordpunct_tokenize ne nécessite aucun
       téléchargement de données NLTK, contrairement à word_tokenize) ;
    3. suppression de la ponctuation et des nombres ;
    4. suppression des mots trop courts ;
    5. suppression des stop words (mots très fréquents, peu informatifs) ;
    6. stemming avec NLTK.

    On utilise la liste de stop words de Gensim pour éviter d'avoir à
    télécharger le corpus `stopwords` de NLTK. NLTK reste bien utilisé pour
    la tokenisation et le stemming, comme demandé dans l'énoncé.
    """

    tokens = wordpunct_tokenize(str(text).lower())

    cleaned = []
    for token in tokens:
        # On ne garde que les suites de lettres (pas de "123", "!", etc.).
        if not token.isalpha():
            continue

        # Un token d'une seule lettre est presque toujours du bruit.
        if len(token) < 2:
            continue

        # Les stop words apparaissent partout et n'aident pas à distinguer les sujets.
        if token in STOPWORDS:
            continue

        cleaned.append(STEMMER.stem(token))

    return cleaned


def load_data():
    """Charge le CSV et supprime les lignes inutilisables.

    On supprime les descriptions vides car elles ne peuvent pas produire de
    vecteur. On réinitialise l'index pour que la position dans les listes
    corresponde bien à la position dans le DataFrame.
    """

    df = pd.read_csv(CSV_PATH)

    df = df.dropna(subset=["description", "article_section"]).copy()
    df["description"] = df["description"].astype(str)
    df["article_section"] = df["article_section"].astype(str).str.strip()

    df = df[df["description"].str.strip() != ""]
    df = df.reset_index(drop=True)

    return df


def build_tfidf(descriptions):
    """Construit le modèle TF-IDF et l'index de similarité.

    Retourne :
    - `vectors` : les vecteurs TF-IDF de chaque article ;
    - `similarity` : l'objet MatrixSimilarity ;
    - `elapsed` : temps de création du modèle, en secondes.

    Le chronomètre démarre avant le préprocessing et s'arrête juste après
    `similarities.MatrixSimilarity(...)`, exactement comme demandé dans
    l'énoncé.
    """

    start = time.perf_counter()

    # 1. Préprocessing de toutes les descriptions.
    texts = [preprocess(description) for description in descriptions]

    # 2. Dictionnaire : associe chaque token à un identifiant entier.
    dictionary = corpora.Dictionary(texts)

    # On enlève les mots qui apparaissent dans moins de 2 documents (bruit)
    # et ceux qui apparaissent dans plus de 90 % des documents (trop généraux).
    dictionary.filter_extremes(no_below=2, no_above=0.9)

    # 3. Bag-of-Words : chaque document devient une liste de (id_token, fréquence).
    bow_corpus = [dictionary.doc2bow(tokens) for tokens in texts]

    # 4. TF-IDF : les mots rares et discriminants prennent plus de poids.
    # `normalize=True` met les vecteurs à une norme de 1, ce qui est la
    # configuration habituelle pour la similarité cosinus.
    tfidf_model = models.TfidfModel(bow_corpus, normalize=True)
    tfidf_vectors = list(tfidf_model[bow_corpus])

    # 5. Index de similarité : c'est l'appel explicitement demandé.
    similarity = similarities.MatrixSimilarity(tfidf_vectors)

    elapsed = time.perf_counter() - start
    return tfidf_vectors, similarity, elapsed


def build_lda(descriptions):
    """Construit le modèle LDA et l'index de similarité.

    Paramètres imposés par l'énoncé :
    - 30 topics ;
    - 2 passes ;
    - random_state=30.

    On filtre un peu plus le dictionnaire que pour TF-IDF : LDA est un modèle
    probabiliste et les mots très rares rendent les topics plus bruités.
    """

    # Comme pour TF-IDF, le chronomètre inclut le préprocessing de ce modèle.
    # On reste ainsi proche de la définition de l'énoncé : "du début du
    # programme jusqu'à l'appel à MatrixSimilarity".
    start = time.perf_counter()

    texts = [preprocess(description) for description in descriptions]

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    bow_corpus = [dictionary.doc2bow(tokens) for tokens in texts]

    # Entraînement de LDA.
    lda_model = models.LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=30,
        passes=2,
        random_state=30,
    )

    # Chaque document est transformé en distribution sur les 30 topics.
    # `minimum_probability=0.0` est important : sinon Gensim supprime les
    # petites probabilités et les vecteurs n'ont plus tous la même dimension.
    lda_vectors = [
        lda_model.get_document_topics(bow, minimum_probability=0.0)
        for bow in bow_corpus
    ]

    similarity = similarities.MatrixSimilarity(lda_vectors)

    elapsed = time.perf_counter() - start
    return lda_vectors, similarity, elapsed


def ratio_quality(similarity_for_query, labels, topic, top_k=TOP_K,
                  exclude_self=False):
    """Implémentation du pseudo-code de l'énoncé.

    `similarity_for_query(i)` doit renvoyer un tableau de scores : le score
    entre l'article `i` et tous les articles du corpus.

    Retourne (ratio_quality, temps_du_pseudo_code, total_goods).

    `exclude_self=False` par défaut : on suit le pseudo-code littéralement.
    L'article est dans le corpus, donc il est son propre voisin le plus
    similaire. Cela gonfle un peu le ratio (au moins 1 bon voisin sur 10).
    Mettre `exclude_self=True` donne une mesure plus réaliste.
    """

    # Articles de la section demandée : ce sont les requêtes.
    query_indices = [
        index
        for index, label in enumerate(labels)
        if label == topic
    ]

    # Sécurité : si la section n'existe pas (par exemple avec un sous-corpus
    # de test), on renvoie zéro au lieu de diviser par zéro.
    if not query_indices:
        return 0.0, 0.0, 0

    total_goods = 0

    # On chronomètre uniquement l'exécution du pseudo-code, pas la création
    # du modèle.
    start = time.perf_counter()

    for query_index in query_indices:
        # Scores de similarité entre la requête et tous les articles.
        scores = np.asarray(similarity_for_query(query_index))

        # argsort trie par ordre croissant ; le signe moins donne l'ordre
        # décroissant (les plus similaires d'abord).
        ranked = np.argsort(-scores)

        # On peut retirer la requête elle-même de ses voisins.
        if exclude_self:
            ranked = ranked[ranked != query_index]

        # Les top_k voisins.
        top_indices = ranked[:top_k]

        # On compte combien de ces voisins sont dans la même section.
        goods = sum(
            1 for index in top_indices
            if labels[int(index)] == topic
        )
        total_goods += goods

    elapsed = time.perf_counter() - start

    # Formule exacte du pseudo-code.
    quality = total_goods / (len(query_indices) * top_k)

    return quality, elapsed, total_goods


def parse_tags(raw_tags):
    """Transforme "Taste ,rice, dessert" en {"taste", "rice", "dessert"}.

    On met en minuscules et on enlève les espaces pour que les tags soient
    comparables même si le CSV contient des espaces irréguliers.
    """

    # Si le champ est vide/NaN, il n'y a aucun tag à comparer.
    if pd.isna(raw_tags):
        return set()

    tags = set()
    for raw_tag in str(raw_tags).split(","):
        tag = raw_tag.strip().lower()
        if tag:
            tags.add(tag)
    return tags


def jaccard(tags_a, tags_b):
    """Similarité de Jaccard entre deux ensembles de tags.

    J(A, B) = |A ∩ B| / |A ∪ B|

    C'est une méthode naturelle pour des tags : elle vaut 1 si les deux
    articles ont exactement les mêmes tags, et 0 s'ils n'en partagent aucun.
    """

    union = tags_a | tags_b
    if not union:
        return 0.0
    return len(tags_a & tags_b) / len(union)


def tag_scores(query_index, tag_sets):
    """Calcule la similarité Jaccard entre un article et tous les autres."""

    query_tags = tag_sets[query_index]
    return np.array([
        jaccard(query_tags, other_tags)
        for other_tags in tag_sets
    ])


def print_topic_results(model_name, labels, similarity_for_query,
                        model_creation_time):
    """Affiche les résultats pour les deux sections demandées."""

    print(f"\n--- {model_name} ---")
    print(f"Temps de création du modèle : {model_creation_time:.2f} s")

    for topic in (FOOD_TOPIC, SPORTS_TOPIC):
        quality, pseudo_time, total_goods = ratio_quality(
            similarity_for_query, labels, topic
        )
        print(
            f"  {topic} : ratio_quality = {quality:.4f} "
            f"(total_goods = {total_goods}, "
            f"pseudo-code = {pseudo_time:.2f} s)"
        )


def main():
    """Point d'entrée du programme."""

    print("Chargement des données...")
    df = load_data()
    descriptions = df["description"].tolist()
    labels = df["article_section"].tolist()

    print(f"{len(df)} articles utilisables")
    print(f"{labels.count(FOOD_TOPIC)} articles 'Food & Drink'")
    print(f"{labels.count(SPORTS_TOPIC)} articles 'Sports'")

    # ------------------------------------------------------------------
    # Tâche 1 : TF-IDF
    # ------------------------------------------------------------------
    print("\n=== Tâche 1 : TF-IDF ===")
    tfidf_vectors, tfidf_similarity, tfidf_time = build_tfidf(descriptions)

    # On passe une petite fonction à ratio_quality pour qu'il puisse récupérer
    # les scores d'un article donné. Cela évite de dupliquer le pseudo-code
    # pour TF-IDF, LDA et les tags.
    print_topic_results(
        "TF-IDF",
        labels,
        lambda index: tfidf_similarity[tfidf_vectors[index]],
        tfidf_time,
    )

    # ------------------------------------------------------------------
    # Tâche 2 : LDA
    # ------------------------------------------------------------------
    print("\n=== Tâche 2 : LDA (30 topics, 2 passes, random_state=30) ===")
    lda_vectors, lda_similarity, lda_time = build_lda(descriptions)

    print_topic_results(
        "LDA",
        labels,
        lambda index: lda_similarity[lda_vectors[index]],
        lda_time,
    )

    # ------------------------------------------------------------------
    # Tâche 3 : Sports vs Food & Drink
    # ------------------------------------------------------------------
    # Les résultats "Sports" sont déjà affichés dans les deux sections
    # précédentes. Il suffit de comparer les deux lignes pour répondre à la
    # question 3 : le ratio Sports est plus faible car le vocabulaire sportif
    # recoupe souvent la politique, le divertissement ou les célébrités.
    print("\n=== Tâche 3 ===")
    print("Comparer les lignes 'Sports' et 'Food & Drink' ci-dessus.")

    # ------------------------------------------------------------------
    # Tâche 4 : similarité par tags (Jaccard)
    # ------------------------------------------------------------------
    print("\n=== Tâche 4 : similarité par tags (Jaccard) ===")
    tag_sets = [parse_tags(tags) for tags in df["tags"]]

    print_topic_results(
        "Tags - Jaccard",
        labels,
        lambda index: tag_scores(index, tag_sets),
        0.0,  # pas de modèle à entraîner, donc pas de temps de création
    )

    print(
        "\nRemarque : les tags sont des métadonnées fiables, mais deux "
        "articles proches peuvent ne partager aucun tag exact. Pour améliorer "
        "les résultats, on pourrait normaliser les synonymes, utiliser une "
        "hiérarchie de tags, ou combiner similarité texte + tags."
    )


if __name__ == "__main__":
    main()
