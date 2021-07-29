"""Nutrient model which calculates a nutrient score
as described:
https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/216094/dh_123492.pdf
Only the "A" points can be calculated at this stage.
To calculate the "C" points, we need knowledge of the fruit,
veg & nuts percentage. More information on the definitions of
fruit, veg and nuts and how it is calculated can be found:
https://webarchive.nationalarchives.gov.uk/20140716083912/http://multimedia.food.gov.uk/multimedia/pdfs/nutprofpguide.pdf
"""
import numpy as np
from ahl_scoping.pipeline.tesco_preprocessing import preprocess_data


def a_points(
    row,
    nutrients=["energyunit", "saturatesunit", "sugarsunit", "saltunit"],
    cutoffs=[
        [0, 335, 670, 1005, 1340, 1675, 2010, 2345, 2680, 3015, 3350, np.inf],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf],
        [0, 4.5, 9, 13.5, 18, 22.5, 27, 31, 36, 40, 45, np.inf],
        [0, 90, 180, 270, 360, 450, 540, 630, 720, 810, 900, np.inf],
    ],
):
    """Return total number of A points from the
    Department of Health's Nutrient Profiling technical
    guidance (Jan 2011)

    Args:
        row: row of dataframe

    Returns:
        int: number of A points, returns -1 if any of the
            nutrient values are NaNs
    """
    a_points = 0
    for nutrient, cutoff in zip(nutrients, cutoffs):
        if np.isnan(row[nutrient]):
            return -1
        for i in range(11):
            if cutoff[i] < row[nutrient] <= cutoff[i + 1]:
                a_points += i
    return a_points


def load_tesco_with_a_points(tesco_groceries_df=preprocess_data()):
    """Returns tesco preprocessed data, with additional column
    for A points from the nutrient profiling model"""
    return tesco_groceries_df.assign(
        a_points=tesco_groceries_df.apply(a_points, axis=1)
    )
