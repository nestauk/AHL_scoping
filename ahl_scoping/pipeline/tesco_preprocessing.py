"""Preprocessing for the tesco grocery data
"""
from ahl_scoping.getters.tesco import get_products, get_categories, get_avg_prices


def join_data():
    """Left join avg price and category dataset
    to the tesco grocery products dataset"""
    return (
        get_products()
        .rename(columns={"x_id": "id"})
        .merge(get_avg_prices().rename(columns={"tpnb": "id"}), on="id", how="left")
        .merge(get_categories().rename(columns={"tpnb": "id"}), on="id", how="left")
    )


def remove_col_prefix(tesco_grocery_df):
    """Remove x_ and y_ prefix from column names"""
    return tesco_grocery_df.rename(columns=lambda x: x.lstrip("x_").lstrip("y_"))


def filter_pet_products(tesco_grocery_df):
    """Filter out pet products"""
    return tesco_grocery_df[tesco_grocery_df["l2y_group"] != "G3 PET PRODUCTS"]


def food_drink(row):
    """Return whether the row is food or drink"""
    if row["category"] in [
        "wine",
        "beer",
        "soft_drinks",
        "water",
        "spirits",
        "tea_coffee",
    ]:
        return "drink"
    else:
        return "food"


def add_food_drink_col(tesco_grocery_df):
    """Add column which indicates whether the product is food or drink"""
    return tesco_grocery_df.assign(
        food_drink=tesco_grocery_df.apply(food_drink, axis=1)
    )


def preprocess_data():
    """Join and preprocess tesco grocery datasets"""
    return (
        join_data()
        .pipe(remove_col_prefix)
        .pipe(filter_pet_products)
        .pipe(add_food_drink_col)
    )
