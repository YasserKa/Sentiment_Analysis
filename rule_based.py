# %% [markdown]
"""
### Rule-based model
Valence Aware Dictionary and sEntiment Reasoner (VADER)
  - Valence score: emotional measurement
  - Lexicon: average of 10 independent human raters [-4, +4]
  - Specialized for Social media
  - Compound: Summing words valence scores; adjusted to rules; normalized between -1 & 1
  - [Paper](https://ojs.aaai.org/index.php/ICWSM/article/view/14550) & [git repository](https://github.com/cjhutto/vaderSentiment/)
"""

# %%

import json

import pandas as pd
from IPython.display import display

# ruff: noqa: E402
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from jupyter_utils import JupyterUtils as JU

DO_PROCESS = False
ju = JU()

# Generated with help of LLMs
# E.g.: cliche, shallow, atmospheric, cringe, wooden, rushed
with open("./movie_lexicon.json", "r", encoding="utf-8") as f:
    extra_lexicon = json.load(f)

analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(extra_lexicon)


sentences = [
    "I like this movie.",
    "I love this movie.",  # Different weights on words
    "I love this movie!",  # Punctuation & emoticons
    "I love this movie! 😍",  # Emojis
    "I LOVE this movie!",  # Capitalization emphasis
    "I really love this movie!!!",  # Repeated punctuation & intensifier
    "I don't love this movie.",  # Negation with contraction
    "This movie is okay.",  # Neutral sentiment
]

results = []

for sentence in sentences:
    score = analyzer.polarity_scores(sentence)
    results.append(
        {
            "Sentence": sentence,
            "Compound Score": score["compound"],
            "Positive": score["pos"],
            "Neutral": score["neu"],
            "Negative": score["neg"],
        }
    )

df = pd.DataFrame(results)
df
# %%
df_reviews = pd.read_csv("./data/raw/rotten_tomatoes_critic_reviews.csv")
df_movies = pd.read_csv("./data/raw/rotten_tomatoes_movies.csv")
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
    """Transform score to [0, 1]"""
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


df_reviews["score_norm"] = df_reviews["review_score"].apply(normalize_score)

df_reviews = df_reviews.dropna(subset=["score_norm"])

# Filter out erranous scores (e.g. 8/5)
df_reviews = df_reviews[df_reviews["score_norm"] <= 1]

# Transform to +ve -ve scores
df_reviews["polarity"] = df_reviews["score_norm"].apply(
    lambda x: 1 if x >= 0.7 else -1 if x <= 0.4 else 0
)

df_reviews = df_reviews[df_reviews["polarity"] != 0]

# %%
ju.freq(df_reviews, "polarity")

# %%
df_reviews = pd.read_csv("./data/processed/rotten_tomatoes_critic_reviews.csv")

# %%

n = 50000  # number you want

pos_samples = df_reviews[df_reviews["polarity"] == 1].sample(n, random_state=42)
neg_samples = df_reviews[df_reviews["polarity"] == 0].sample(n, random_state=42)

df_reviews_ = pd.concat([pos_samples, neg_samples], ignore_index=True)


# %%
# %%execute_if DO_PROCESS
import spacy
from spellchecker import SpellChecker

nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()


def clean_text(text: str) -> str:
    """Correct typos"""
    tokens = nlp(text)
    cleaned_tokens = []

    for token in tokens:
        if token.is_alpha:
            # Remove 3 or more repeated characters, to improve chances of correcting words
            word = re.sub(r"(.)\1{2,}", r"\1\1", token.text)
            corrected = spell.correction(word)
            corrected = corrected if corrected else word
            cleaned_tokens.append(corrected + token.whitespace_)
        else:
            cleaned_tokens.append(token.text_with_ws)

    return "".join(cleaned_tokens)


df_reviews["clean_review"] = df_reviews["review_content"].progress_apply(clean_text)

df_reviews.to_csv(
    "./data/processed/rotten_tomatoes_critic_reviews_rule_based.csv", index=False
)

# %%

df_ = pd.merge(
    df_movies[["movie_title", "rotten_tomatoes_link"]],
    df_reviews,
    left_on="rotten_tomatoes_link",
    right_on="rotten_tomatoes_link",
    how="inner",
)

import json
import re

# %%
from collections import Counter

from tqdm import tqdm

tqdm.pandas()  # enables df.progress_apply()


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# Remove full movie title from text
def remove_movie_title(text, title):
    # Case-insensitive removal
    return re.sub(re.escape(title), "", text, flags=re.IGNORECASE).strip()


def vader_sentiment_with_word_counts(text, title):
    clean_text = remove_movie_title(text, title)
    words = tokenize(clean_text)
    pos_counter, neg_counter = Counter(), Counter()
    for word in words:
        if word in analyzer.lexicon:
            score = analyzer.lexicon[word]
            if score > 0:
                pos_counter[word] += 1
            elif score < 0:
                neg_counter[word] += 1
    scores = analyzer.polarity_scores(text)
    return pd.Series(
        {
            "compound": scores["compound"],
            "pos_words": pos_counter,
            "neg_words": neg_counter,
        }
    )


df_ = df_.dropna(subset="review_content")
# Apply per review
df_sentiment = df_.join(
    df_.progress_apply(
        lambda x: vader_sentiment_with_word_counts(
            x["review_content"], x["movie_title"]
        ),
        axis=1,
    )
)

# %%
# For neutral
# df_sentiment["pred_polarity"] = df_sentiment["compound"].apply(
#     lambda x: 1 if x >= 0.1 else -1 if x <= -0.1 else 0
# )
df_sentiment["pred_polarity"] = df_sentiment["compound"].apply(
    lambda x: 1 if x >= 0 else -1
)


def merge_counters(counters):
    result = Counter()
    for c in counters:
        result.update(c)
    return result.most_common(5)


agg_df = (
    df_sentiment.groupby("movie_title")
    .agg(
        num_reviews=("review_content", "count"),
        mean_compound=("compound", "mean"),
        pos_words=("pos_words", merge_counters),
        neg_words=("neg_words", merge_counters),
        mean_polarity=("polarity", "mean"),
        mean_pred_polarity=("pred_polarity", "mean"),
        accuracy_polarity=(
            "pred_polarity",
            lambda x: (x == df_sentiment.loc[x.index, "polarity"]).mean(),
        ),  # per-movie accuracy
    )
    .reset_index()
)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

df_sentiment = df_sentiment.dropna(subset=["pred_polarity"])
# Overall metrics
y_true = df_sentiment["polarity"]
y_pred = df_sentiment["pred_polarity"]

accuracy = (y_true == y_pred).mean()
cm = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

# Heatmap for confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Neg", "Pos"],
    yticklabels=["Neg", "Pos"],
)
plt.title("Multiclass Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-score:  {f1:.2f}")


# %%
with ju.full_display():
    display(
        agg_df.sort_values(by="num_reviews", ascending=False)
        .reset_index(drop=True)
        .head()
    )

# %%
with ju.full_display():
    display(
        df_sentiment[df_sentiment["polarity"] != df_sentiment["pred_polarity"]][
            [
                "movie_title",
                "review_score",
                "review_content",
                "score_norm",
                "polarity",
                "compound",
                "pos_words",
                "neg_words",
                "pred_polarity",
            ]
        ].head()
    )
# %% [markdown]
"""
### Limitations

  - Doesn't understand context, idioms, sarcasm, irony, etc.
  - Intensive manual labor
  - Unscalable with many rules
"""
