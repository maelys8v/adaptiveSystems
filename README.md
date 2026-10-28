# Adaptive Systems assignment

This project contains the experiments for Tasks 1-4 and a Streamlit news
recommender prototype for Task 5. Both use the supplied `news2.csv` dataset.

## 1. Install Python and get the project

Install **Python 3.11** (the version used to test this project), including pip,
from [python.org](https://www.python.org/downloads/). On Windows, select
**Add Python to PATH** during installation, then reopen your terminal.
On Linux, Python and pip can also be installed through your distribution's
package manager. Use Python 3.11 for the commands below; newer Python versions
may not have compatible Gensim packages.

Download and extract the project archive. Keep `news2.csv` alongside `main.py`
and `prototype.py`. Open a terminal in that folder (usually `adaptiveSystems`).
No editor or Git installation is required to run the application.

Check that Python and pip are available:

```sh
python --version
python -m pip --version
```

On macOS/Linux, use `python3.11` instead of `python` in all commands below if
that is the name of your Python 3.11 executable.
If you use a virtual environment, activate it before installing dependencies
and keep it active when running the application.

## 2. Install the dependencies

Run this command from the project folder. It installs **all** required packages,
including Streamlit, Gensim, NLTK, NumPy, SciPy and pandas:

```sh
python -m pip install -r requirements.txt
```

An internet connection is needed for this installation. No separate NLTK corpus
download, account, database or news API is required.

## 3. Run Task 5: news recommender prototype

```sh
python -m streamlit run prototype.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).
The first run loads the articles and normalizes their tags; subsequent interactions
reuse this catalogue. No TF-IDF or LDA model is trained by the interface.
Keep the terminal open while using the application. Press **Ctrl+C** in the
terminal to stop the server. To launch it again, rerun the same command.
After dependency installation, the application uses only the local CSV.

If Python reports `No module named streamlit`, rerun the dependency installation
with the same Python executable used to launch the app. If the CSV cannot be
found, check that `news2.csv` is next to `prototype.py`.

## Four read-only pages

The application illustrates Alex, a fictional reader whose preferences and history
are fixed. Only navigation is active; scenario data cannot be edited.

1. **Initial preferences:** Sports, Tech and Food & Drink displayed in a disabled
   selection control. Start reading opens the feed.
2. **Recommended feed:** three predefined real articles covering basketball,
   online privacy and pumpkin recipes, with section explanations and disabled
   feedback buttons. Open description navigates to the article.
3. **Article details:** the real description, section and tags, a predefined
   feedback state and related articles based on Jaccard tag similarity.
4. **My interests:** read-only initial sections, example interests, and fictional
   history with 2 likes, 2 rejections and 4 opened articles.

Use the sidebar to navigate directly to each page. Article details in the sidebar
always opens the basketball example. Related links and the profile's revisit
shortcut only change the article being viewed. They never record feedback.

## What is illustrative and what works

Headlines, descriptions, sections and tags are real CSV values. Alex, the selected
preferences and all interactions are fictional. The feed is curated, not learned.
Highlighted Like / Not Interested states are predefined and disabled. Preference
controls and Save preferences are also disabled. There is no editable profile,
feedback capture, reading-history capture or adaptive ranking, and no display-mode
switch, reset control or pagination.

Related articles use actual item-to-item Jaccard similarity via `parse_tags()` and
`tag_scores()` from `main.py`. Importing it does not run the Tasks 1-4 experiments.
Only navigation state is kept locally; previous editable state is discarded.

## Written response and page references

[`practical_assignment_1.md`](practical_assignment_1.md) is the final report for
Tasks 1-5. Task 5 covers the representation choice, proposed user model,
recommendation logic, prototype navigation guide, four page descriptions and
limitations. The text uses **See prototype page 1/2/3/4** references to identify
the corresponding application screens.

The adaptive behavior is a written proposal and has not been evaluated as a
personalized recommender. The CSV has no publication
dates or full article bodies; no freshness scores or full-text articles are
invented. No TF-IDF or LDA model is trained by the interface.

Automated acceptance checks use the real CSV and Streamlit's
[AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest):

```sh
python -m unittest test_prototype
```

## Run Tasks 1-4

After installing the dependencies above, run the original experiments from the
folder containing `main.py` and `news2.csv`:

```sh
python main.py
```

`main.py` is the English version, including comments and output messages.
`main_fr.py` preserves the original French version.

The same `requirements.txt` is used for Tasks 1-4 and the Task 5 interface.
