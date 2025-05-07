import numpy as np


def data_preprocessing(data):

    fix_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in fix_cols:
        data[col] = data[col].replace(0, np.nan)
        median = data[col].median()
        data[col] = data[col].fillna(median)

    return data
