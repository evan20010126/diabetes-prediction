import numpy as np


def remove_outliers_iqr(series):
    """
    Remove outliers from a series using the IQR method.
    """

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series[(series >= lower) & (series <= upper)]


def impute_missing_values(data):
    """
    Impute missing values in the dataset by replacing zeros with NaN,
    removing outliers temporarily, and filling missing values with the median.

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with imputed values.
    """

    fix_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for col in fix_cols:
        # Replace zeros with NaN
        data[col] = data[col].replace(0, np.nan)

        # Remove outliers temporarily and calculate median
        clean_series = remove_outliers_iqr(data[col].dropna())
        median = clean_series.median()

        # Fill missing values with the calculated median
        data[col] = data[col].fillna(median)

    return data
