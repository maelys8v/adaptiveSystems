# Practical Assignment 1

**Adaptive Systems**

Group: Maelys DEVAUD, Edgar BARRIOS, Numidia NIMHAOULIN

## Task 1: TF-IDF

For the implementation of the pseudocode, see [main.py](main.py).

The model creation time is 18.58 s, and the pseudocode execution time for Food & Drink is 1.69 s.

| Method | Category | Model creation time (s) | Pseudocode execution time (s) | `total_goods` | `ratio_quality` |
| --- | --- | ---: | ---: | ---: | ---: |
| TF-IDF | Food & Drink | 18.58 | 1.69 | 771 | 0.6320 |

## Task 2: LDA and comparison with TF-IDF

LDA uses 30 topics, 2 passes, and `random_state=30`.

| Method | Category | Model creation time (s) | Pseudocode execution time (s) | `total_goods` | `ratio_quality` |
| --- | --- | ---: | ---: | ---: | ---: |
| TF-IDF | Food & Drink | 18.58 | 1.69 | 771 | 0.6320 |
| LDA | Food & Drink | 23.55 | 0.04 | 875 | 0.7172 |

For the execution times, we go from 18.58 s for TF-IDF to 23.55 s for LDA for model creation. For the pseudocode, we go from 1.69 s for TF-IDF to 0.04 s for LDA. So we can say that LDA takes a little more time to create the model, but the pseudocode execution time is much shorter, making it interesting for fast similarity search.

The reason for the large gap in pseudocode execution time is that TF-IDF creates a vector for each article representing the importance of each word in it compared to the rest of the corpus, with one dimension per word in the filtered vocabulary. The similarity index stores these vectors in a dense representation, which is still much larger than the representation used for LDA. For LDA, we only have a 30-dimensional vector, where each component represents the probability associated with one of the 30 learned topics. Computing cosine similarity between 30-dimensional vectors is therefore much faster than between vectors with hundreds or thousands of dimensions with TF-IDF.

The article vectors are computed during model creation. The pseudocode execution time measures similarity search and evaluation using those precomputed vectors; it does not include inferring the topic distribution of a new article.

For the results, we obtain 0.6320 for TF-IDF and 0.7172 for LDA, an increase of 8.52 percentage points for LDA compared to TF-IDF. LDA may perform better for Food & Drink because it can assign similar topic distributions to articles that use different words associated with the same topics, whereas TF-IDF relies on shared terms after preprocessing.

## Task 3: Sports compared with Food & Drink

| Method | Category | Usable articles | `total_goods` | `ratio_quality` | Pseudocode execution time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| TF-IDF | Food & Drink | 122 | 771 | 0.6320 | 1.69 |
| TF-IDF | Sports | 110 | 523 | 0.4755 | 1.47 |
| LDA | Food & Drink | 122 | 875 | 0.7172 | 0.04 |
| LDA | Sports | 110 | 389 | 0.3536 | 0.04 |

Model creation takes 18.58 s for TF-IDF and 23.55 s for LDA. Each model is built once on the corpus and then evaluated separately for the two categories.

For both TF-IDF and LDA, we observe lower `ratio_quality` scores for Sports than for Food & Drink. Several factors may explain this difference. First, some Sports articles mention public figures associated with other categories. Examples in the dataset include Barack Obama attending a Chicago Bulls game and Prince William reacting to a football victory. These articles therefore contain vocabulary related to politics or celebrities, which may increase their similarity to articles outside the Sports category.

Second, Sports covers diverse disciplines, including basketball, cricket, gymnastics and horse racing. Each discipline has its own vocabulary, so two articles belonging to Sports may share relatively few words.

For TF-IDF, IDF reduces the weight of frequent words and gives greater weight to less frequent words. Consequently, public figures' names may strongly influence similarity when they are relatively rare in the corpus and shared by several articles. This can favour neighbours mentioning the same person over neighbours discussing other sports.

For LDA, the 30 learned topics do not necessarily correspond to the dataset's editorial categories. Sports articles may have different topic distributions depending on their discipline or their political and social content. Consequently, some may be closer to articles from other categories than to other Sports articles. The two categories also have relatively similar numbers of usable articles: 122 for Food & Drink and 110 for Sports. Although this difference may contribute to the results, the counts alone do not establish that it explains the observed gap.

Finally, Food & Drink articles may share a more consistent culinary vocabulary, including ingredients, cooking techniques and recipe terms. Although they also mention countries and other proper names, the surrounding culinary vocabulary may help maintain similarity within the category. This is not an absolute rule, since food articles can also overlap with health, culture or environmental issues.

Overall, the lower `ratio_quality` for Sports may reflect greater overlap with other categories, more varied vocabulary across disciplines, and the influence of shared public figures' names. In contrast, the recurring culinary vocabulary in Food & Drink may make articles within that category easier to retrieve together. These are plausible explanations rather than causes established by the scores alone.

### Evaluation convention

The evaluation uses the 10 most similar articles for each query. `total_goods` counts retrieved articles belonging to the query category, and `ratio_quality = total_goods / (number of queries × 10)`. The query article itself remains eligible among the retrieved results, following the implementation's default setting (`exclude_self=False`).

## Task 4 – Tag-based Similarity

To improve article similarity, we can use the tags already associated with each article instead of relying only on their textual descriptions. Tags provide a more structured representation of an article's main topics.

We propose using **Jaccard similarity**, which measures the proportion of shared tags between two articles:

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

where A and B represent the sets of tags associated with two articles.

We evaluated this approach using the same `ratio_quality` metric as in the previous tasks.

| Method | Food & Drink | Sports |
|---|---:|---:|
| TF-IDF (descriptions) | 0.6320 | 0.4755 |
| LDA (descriptions) | **0.7172** | 0.3536 |
| Jaccard (tags) | 0.6631 | **0.5664** |
| TF-IDF + Cosine (tags) | 0.6164 | 0.5364 |

**Discussion:** Jaccard improves the results for Sports compared with both TF-IDF and LDA, suggesting that tags can better capture the main subject of sports articles. However, for Food & Drink, LDA performs better, possibly because it captures broader thematic relationships beyond exact tag matches.

We also experimented with **TF-IDF weighting on tags**, but it performed slightly worse than Jaccard for both categories. One possible explanation is that TF-IDF reduces the importance of frequently occurring tags, even though these tags may be useful for identifying articles belonging to the same general category. While TF-IDF is well suited to identifying discriminative words in textual descriptions, this weighting may be less appropriate for tags when the objective is to retrieve articles from the same broad category.

Overall, **tagging can improve article similarity, but its effectiveness depends on the category and the similarity measure used**. A limitation of Jaccard is that it relies on exact tag matches and cannot recognize synonyms such as "football" and "soccer" without additional normalization.

## Task 5: Content-Based News Recommender System

### 1. Proposed recommender and choice of representation

We propose a **content-based news recommender** that uses article tags and the
reader's interests. It does not require ratings from other users. This approach
reuses the tag-similarity technique investigated in Task 4.

Tags are split on commas, trimmed, converted to lowercase and deduplicated.
Multiword tags remain complete labels. We compare two articles using **Jaccard
similarity** between their tag sets:

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

If both sets are empty, similarity is zero.

We chose Jaccard because it is simple to interpret and gives balanced results
across Food & Drink and Sports in our experiments. Compared with LDA, it performs
much better on Sports while scoring only slightly lower on Food & Drink. We
therefore propose reusing it to compare candidate articles with articles the
reader liked or rejected, linking content similarity to the user profile.
These results support our choice of similarity measure; they do not establish
that it produces the best personalized recommendations.

### 2. Information stored in the proposed user model

The user model records the reader's interests and interactions. For a reader
such as Alex, we would store:

| What we store | Example for Alex | Purpose |
| --- | --- | --- |
| Initial categories | Sports, Tech, Food & Drink | Suggest relevant articles before any interactions. |
| Liked article IDs | A basketball article and a pumpkin-recipe article | Find articles with similar tags. |
| Rejected article IDs | An Entertainment article and a Politics article | Reduce recommendations with similar tags. |
| Opened article IDs | The two liked articles, a Tech article and the rejected Entertainment article | Track articles already seen and identify weaker interest signals. |

Article IDs link to the tags already stored in the dataset. A Like is stronger
evidence than an opening, and Not Interested takes priority over an opening of
the same article.

### 3. How recommendations would be generated

For a new reader, the system would recommend articles from the categories chosen
at the start. Without selected categories, it would offer a general selection.

After interactions, the system would compare candidate articles with those in
the reader's history using Jaccard similarity. Similarity to liked articles would
increase a candidate's score, while similarity to rejected articles would decrease
it. Opening an article without rating it would provide a weaker positive signal.
Articles would be ranked by the combined score, excluding articles already seen.

For example, liking a basketball article would promote other articles sharing
its tags. New feedback would update subsequent recommendations. This is the
**proposed algorithm**; the prototype below illustrates the screens and user
model without implementing personalized ranking or feedback updates.

### 4. Low-fidelity prototype and illustrated scenario

The Streamlit prototype contains **four read-only pages** for a fictional reader,
Alex. It uses real articles from `news2.csv`, with predefined preferences,
feedback and history. Alex's initial categories are Sports, Tech and Food & Drink;
the example profile contains **2 likes, 2 rejections and 4 opened articles**.

The feed is predefined, and preference and feedback controls are disabled.
Navigation is active. The Related articles feature computes actual item-to-item
Jaccard similarity, independently of Alex's example profile.

**Prototype navigation guide.** From the `adaptiveSystems` directory, run
`python -m streamlit run prototype.py` and open the local URL shown. Use the
sidebar to visit the four pages referenced below.

| Report reference | Sidebar page | What it illustrates |
| --- | --- | --- |
| Page 1 | Initial preferences | Alex's selected categories. |
| Page 2 | Recommended feed | Three example articles with feedback controls. |
| Page 3 | Article details | An article description and Jaccard-related articles. |
| Page 4 | My interests | Alex's interests, feedback and reading history. |

#### 4.1 Initial preferences

**See prototype page 1 - Initial preferences.**

This page displays Sports, Tech and Food & Drink as Alex's selected categories.
In the proposed system, readers would choose these categories to initialize their
profile. The Start reading button opens the example feed.

#### 4.2 Recommended feed

**See prototype page 2 - Recommended feed.**

The feed contains three predefined articles about basketball, online privacy and
pumpkin recipes. Each card shows the headline, category, description snippet,
tags and a category-based explanation. Open description navigates to the article;
Like and Not Interested illustrate explicit feedback. The basketball and recipe
articles have predefined Liked states.

#### 4.3 Article details and feedback

**See prototype page 3 - Article details.**

The sidebar opens **"High School Basketball Player Hits Ridiculous Game-Winning
Shot"**, which also appears in the feed and Alex's history. The page displays its
description, category, tags and predefined Liked state. The dataset provides
article descriptions rather than full article text.

Related articles are selected using Jaccard similarity between tags and can be
opened. Back to feed returns to the example feed. These actions only navigate;
they do not record interactions.

#### 4.4 My interests and user history

**See prototype page 4 - My interests.**

This page displays Alex's selected categories, an illustrative interest summary,
and lists of liked, rejected and opened articles. The counters show **2 likes,
2 rejections and 4 opened articles**. Sports and Food & Drink reflect the example
likes, while Tech is an initial preference. This screen illustrates the user
model described in Section 2; its contents are fixed in the prototype.

### 5. Strengths and limitations

The approach is simple to explain and can recommend tagged articles without
other users' ratings. Its main limitations are missing tags and exact matching:
Jaccard does not automatically recognize synonyms such as football and soccer.
The prototype illustrates the proposed interface; evaluating personalized
recommendation quality would require testing the proposed algorithm with users.