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
from ahl_scoping.pipeline.nutrient_model import load_tesco_with_a_points
import seaborn as sns

# %%
tesco_groceries = load_tesco_with_a_points()

# %%
# plot distribution of a_points
sns.histplot(data=tesco_groceries, x="a_points", bins=26)

# %%
# plot distribution of a_points without -1 values (-1 is given when one of the nutrient values is NaN)
sns.histplot(
    data=tesco_groceries[tesco_groceries["a_points"] != -1], x="a_points", bins=26
)

# %%
# plot energy unit against avg price, coloured by a points
food_only = tesco_groceries[tesco_groceries["food_drink"] == "food"]
sns.set(rc={"figure.figsize": (16, 12)})
sns.scatterplot(data=food_only, x="avg_price", y="energyunit", hue="a_points")

# %%
# plot energy unit against a points
food_only = tesco_groceries[tesco_groceries["food_drink"] == "food"]
sns.set(rc={"figure.figsize": (16, 12)})
sns.scatterplot(data=food_only, x="avg_price", y="a_points")

# %%
