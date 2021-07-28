# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from ahl_scoping.utils.file_management import load_pickle
from ahl_scoping import PROJECT_DIR
from ahl_scoping.pipeline.tesco_clustering import (
    FILEPATH_LDA_OUTPUT,
    FILEPATH_LDA_MODEL,
    FILEPATH_TFIDF_INGREDIENTS,
    FILEPATH_TFIDF_VECTORISER,
)
from ahl_scoping.pipeline.tesco_preprocessing import preprocess_data
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from itertools import chain

# %%
lda_output = load_pickle(FILEPATH_LDA_OUTPUT)
lda_model = load_pickle(FILEPATH_LDA_MODEL)
tfidf_ingredients = load_pickle(FILEPATH_TFIDF_INGREDIENTS)
tfidf_vectoriser = load_pickle(FILEPATH_TFIDF_VECTORISER)

# %%
# load tesco data and drop rows without ingredients
tesco_groceries = preprocess_data().dropna(subset=["ingredients"])

# %%
# assign most related lda topic to items in dataset
tesco_groceries["lda_assignment"] = lda_output.argmax(axis=1)

# %%
# add column for top 3 tfidf ingredients
sorted_tfidf_indices = [indices[-3:] for indices in np.argsort(tfidf_ingredients)]

feature_names = np.asarray(tfidf_vectoriser.get_feature_names())
tesco_groceries["top3_tfidf_ingredients"] = [
    list(feature_names[idxs]) for idxs in sorted_tfidf_indices
]


# %%
def word_cloud(words, title):
    wc = WordCloud(
        background_color="white", max_words=40, colormap="Dark2", max_font_size=170
    )
    wc.generate(" ".join(list(chain(*list(words)))))
    plt.figure(figsize=(16, 13))
    plt.title(title, fontsize=20)
    plt.imshow(wc.recolor(colormap="plasma", random_state=0))
    plt.axis("off")


# %%
# view wordcloud for group
group = 2
words = tesco_groceries[
    tesco_groceries["lda_assignment"] == group
].top3_tfidf_ingredients.values
word_cloud(words, title=f"wordcloud for group {group}")

# %%
# see counts of item subclass in group
tesco_groceries[
    tesco_groceries["lda_assignment"] == group
].l5y_subclass.value_counts().head(20)

# %%
# see counts of item class in group
tesco_groceries[
    tesco_groceries["lda_assignment"] == group
].l4y_class.value_counts().head(20)

# %%
