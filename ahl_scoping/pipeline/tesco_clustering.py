"""Clustering for the tesco grocery data
"""
from ahl_scoping.pipeline.tesco_preprocessing import preprocess_data
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from ahl_scoping.utils.file_management import save_pickle
from ahl_scoping import PROJECT_DIR

PATH_MODELS = PROJECT_DIR / "outputs/models/"
FILEPATH_LDA_OUTPUT = PATH_MODELS / "lda_output.pkl"
FILEPATH_LDA_MODEL = PATH_MODELS / "lda_model.pkl"
FILEPATH_TFIDF_INGREDIENTS = PATH_MODELS / "tfidf_ingredients.pkl"
FILEPATH_TFIDF_VECTORISER = PATH_MODELS / "tfidf_vectoriser.pkl"

N_LDA_COMPONENTS = 10


def remove_phrases(ingredients_series):
    """Remove items in ingredients list that are not ingredients
    or that are repititions of items"""
    return (
        ingredients_series.str.replace("||", "|", regex=False)
        .str.split("|")
        .apply(lambda x: [item.lower() for item in x])
        .apply(
            lambda x: [
                item
                for item in x
                if "contain" not in item
                if "percent" not in item
                if "may contain" not in item
                if "may also contain" not in item
                if "prepared" not in item
                if "made in a" not in item
                if "produced" not in item
                if " used " not in item
                if " use " not in item
                if "activity" not in item
                if "moisture" not in item
                if "product" not in item
                if "ingredients" not in item
                if " is a " not in item
            ]
        )
        .apply(lambda x: [item.split(":")[0] for item in x])
        .str.join(", ")
    )


def remove_formatting(ingredients_series):
    """Remove text formatting and non alphabetic characters,
    replace . with ,"""
    return (
        ingredients_series.str.replace("<strong>", "", regex=False)
        .str.replace("</strong>", "", regex=False)
        .str.replace(" \(.*?\)", "", regex=True)
        .str.replace(" \<.*?\>", "", regex=True)
        .str.replace(" \{.*?\}", "", regex=True)
        .str.replace(" \[.*?\]", "", regex=True)
        .str.replace("[^a-zA-Z\s,:-]+", "", regex=True)
        .str.replace(".", ",", regex=False)
    )


def add_underscores(ingredients_series):
    """Add underscores between each ingredient
    and remove measurement terms"""
    return ingredients_series.str.split(", ").apply(
        lambda x: [
            item.replace(" ", "_")
            .replace(")", "")
            .replace("-", "_")
            .replace("___", "_")
            .replace("__", "_")
            .replace("a_bit_of_", "")
            .replace("a_blend_of_", "")
            .replace("a_chunk_of_", "")
            .replace("_in_variable_proportions", "")
            .replace("_in_varying_proportions", "")
            .replace("_mgkg", "")
            .replace("some_", "")
            .replace("a_dash_of_", "")
            .replace("a_crushed_", "")
            for item in x
        ]
    )


def remove_duplicates(ingredients_series):
    """Remove duplicate ingredients, remove trailing
    underscores, remove items containing '_g_'
    and replace emulsifier descriptions with 'emulsifier'
    """
    return (
        ingredients_series.apply(
            lambda x: ["emulsifier" if "emulsifier" in item else item for item in x]
        )
        .apply(lambda x: [item.rstrip("_").lstrip("_") for item in x])
        .apply(lambda x: [item for item in x if "_g_" not in item])
        .map(set)
        .str.join(", ")
    )


def remove_unique_ingredients(ingredients_series):
    """Remove ingredients from ingredients series
    if that ingredient only appears once in all
    the ingredients lists"""
    ings_split = [ing.split(", ") for ing in ingredients_series]
    freq = defaultdict(int)
    for ingredients in ings_split:
        for ingredient in ingredients:
            freq[ingredient] += 1

    return (
        ingredients_series.str.split(", ")
        .apply(lambda x: [ingredient for ingredient in x if freq[ingredient] > 1])
        .apply(lambda x: ", ".join([i for i in x]))
    )


def reformat_ingredients(ingredients_series):
    """Reformat ingredients lists in ingredients series
    into the format: item_1, item_2, item_3
    """
    return (
        ingredients_series.pipe(remove_phrases)
        .pipe(remove_formatting)
        .pipe(add_underscores)
        .pipe(remove_duplicates)
        .pipe(remove_unique_ingredients)
    )


def process_ingredients_for_clustering():
    """Return list of ingredient lists"""
    return list(
        reformat_ingredients(
            preprocess_data().dropna(subset=["ingredients"])["ingredients"]
        )
    )


def vectorise_ingredients():
    """Returns TFidf vectorised ingredients lists and vectorizer"""
    vectorizer = TfidfVectorizer()
    return (
        vectorizer.fit_transform(process_ingredients_for_clustering()).toarray(),
        vectorizer,
    )


def lda_transform(vectorised_ingredients, n_components):
    """Returns array of LDA topic assignments and LDA model"""
    lda_model = LatentDirichletAllocation(
        n_components=n_components, random_state=0, n_jobs=-1
    )
    return lda_model.fit_transform(vectorised_ingredients), lda_model


if __name__ == "__main__":
    tfidf_ingredients, vectoriser = vectorise_ingredients()
    lda_output, lda_model = lda_transform(tfidf_ingredients, N_LDA_COMPONENTS)
    save_pickle(lda_output, FILEPATH_LDA_OUTPUT)
    save_pickle(lda_model, FILEPATH_LDA_MODEL)
    save_pickle(tfidf_ingredients, FILEPATH_TFIDF_INGREDIENTS)
    save_pickle(vectoriser, FILEPATH_TFIDF_VECTORISER)
