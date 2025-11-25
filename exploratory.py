# %%
# ruff: noqa: E402
import pandas as pd

from jupyter_utils import JupyterUtils as JU

ju = JU()
DO_PROCESS = False

# %%
df_reviews = pd.read_csv("./data/raw/rotten_tomatoes_critic_reviews.csv")
df_movies = pd.read_csv("./data/raw/rotten_tomatoes_movies.csv")

# %%
# Remove reviews with null scores or content
df_reviews = df_reviews.dropna(subset=["review_score", "review_content"])

# %%
from fractions import Fraction

import numpy as np

# Normalize scores to 0-1
GRADES = {
    "A+": 12,
    "A": 11,
    "A−": 10,
    "B+": 9,
    "B": 8,
    "B−": 7,
    "C+": 6,
    "C": 5,
    "C−": 4,
    "D+": 3,
    "D": 2,
    "D−": 1,
    "F": 0,
}


def normalize_score(score):
    """docstring for convert_score"""
    if "/" in score:
        try:
            num, den = score.split("/")
            num = float(Fraction(num))
            den = float(Fraction(den))
            if den > 0:
                return num / den
            else:
                return np.nan
        except Exception:
            return np.nan

    # Remove white spaces
    score = score.replace(" ", "")

    # Letter grade
    if score in GRADES:
        return GRADES[score] / 12

    # Some values are numeric without "/", ignore them
    return np.nan


df_reviews_ = df_reviews.copy()

df_reviews_["score_norm"] = df_reviews["review_score"].apply(normalize_score)

df_reviews_ = df_reviews_.dropna(subset=["score_norm"])

# Filter out erranous scores (e.g. 8/5)
df_reviews_ = df_reviews_[df_reviews_["score_norm"] <= 1]
# %%
ju.freq(df_reviews_, "score_binary")


# %%
# Transform to +ve -ve scores
df_reviews_["score_binary"] = df_reviews_["score_norm"].apply(
    lambda x: 1 if x > 0.5 else 0
)

# %%
n = 50000  # number you want

pos_samples = df_reviews_[df_reviews_["score_binary"] == 1].sample(n, random_state=42)
neg_samples = df_reviews_[df_reviews_["score_binary"] == 0].sample(n, random_state=42)

df_reviews__ = pd.concat([pos_samples, neg_samples], ignore_index=True)

# %%
# %%execute_if DO_PROCESS
import spacy
from tqdm import tqdm

tqdm.pandas()  # enables df.progress_apply()

# Download the English model
# python -m space en_core_web_sm
nlp = spacy.load("en_core_web_sm")


# Remove stopwords and
def preprocess_review(text):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha
    ]
    return " ".join(tokens)


df_reviews__["preproced_review"] = df_reviews__["review_content"].progress_apply(
    preprocess_review
)

df_reviews__.to_csv("./data/processed/rotten_tomatoes_critic_reviews.csv", index=False)


# %%

df_reviews__ = pd.read_csv("./data/processed/rotten_tomatoes_critic_reviews.csv")

# %%
ju.vz(df_reviews__)

# %%
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# Remove reviews with null scores or content
df_reviews__ = df_reviews__.dropna(subset=["preproced_review"])
X = df_reviews__["preproced_review"]
y = df_reviews__["score_binary"]


kf = KFold(n_splits=5, shuffle=True, random_state=42)

tv = TfidfVectorizer(ngram_range=(1, 3))

cv = CountVectorizer(ngram_range=(1, 3))

fold_results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    X_train_bow = tv.fit_transform(X_train)
    X_test_bow = tv.transform(X_test)

    # clf = LogisticRegression(max_iter=200)
    lr = LogisticRegression(penalty="l2", max_iter=500, C=1, random_state=42)
    lr.fit(X_train_bow, y_train)

    preds = lr.predict(X_test_bow)
    acc = accuracy_score(y_test, preds)

    fold_results.append(acc)

print("Accuracy per fold:", fold_results)
print("Mean accuracy:", sum(fold_results) / len(fold_results))


# %%
# Visualization

df_ = pd.merge(
    df_reviews,
    df_movies,
    left_on="rotten_tomatoes_link",
    right_on="rotten_tomatoes_link",
    how="inner",
)


df__ = (
    df_.groupby("rotten_tomatoes_link")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
print(df__)

# %%

df = df_reviews.copy()
df = df_reviews[
    (df_reviews["rotten_tomatoes_link"] == "m/star_wars_the_rise_of_skywalker")
    | (df_reviews["rotten_tomatoes_link"] == "m/solo_a_star_wars_story")
    | (df_reviews["rotten_tomatoes_link"] == "m/star_wars_the_last_jedi")
]

# %%
from collections import Counter

reviews = [
    "This movie was fantastic and thrilling",
    "I did not like the plot but loved the visuals",
]


def get_frequent_words(reviews, top_n=10):
    words = []
    for review in reviews:
        doc = nlp(review)
        # Keep only alphabetic words that are not stop words
        words.extend(
            [
                token.text.lower()
                for token in doc
                if not token.is_stop and token.is_alpha
            ]
        )
    freq = Counter(words)
    return freq.most_common(top_n)


# Group by movie and compute most frequent words
result = (
    df_[:10000]
    .groupby("movie_title")["review_content"]
    .apply(lambda x: get_frequent_words(x, top_n=5))
)
print(result)
