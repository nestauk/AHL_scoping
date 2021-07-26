"""Data getters for the tesco data.

Data source: https://figshare.com/collections/Tesco_Grocery_1_0/4769354/2
"""
import pandas as pd
from ahl_scoping import PROJECT_DIR

TESCO_DIR = PROJECT_DIR / "inputs/tesco_data"


def get_products() -> pd.DataFrame:
    """Returns dataframe of tesco products"""
    return pd.read_csv(TESCO_DIR / "food_products.csv")


def get_categories() -> pd.DataFrame:
    """Returns dataframe of tesco food categories"""
    return pd.read_csv(TESCO_DIR / "food_categories.csv")


def get_avg_prices() -> pd.DataFrame:
    """Returns dataframe of tesco average prices"""
    return pd.read_csv(TESCO_DIR / "product_avgprices.csv")
