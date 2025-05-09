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

    # All of attributes: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome]
    fix_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for col in fix_cols:
        # Replace zeros with NaN
        data[col] = data[col].replace(0, np.nan)

        # Remove outliers temporarily and calculate median
        clean_series = remove_outliers_iqr(data[col].dropna())

        skew = clean_series.skew()
        if abs(skew) >= 0.5:
            # If skewness is high, use the median of the original series
            fill_value = data[col].median()
        else:
            # If skewness is low, use the mean of the cleaned series
            fill_value = clean_series.mean()

        # Fill missing values with the calculated median
        data[col] = data[col].fillna(fill_value)

    data = add_combined_features(data)
    return data


def add_combined_features(data):

    epsilon = 1e-5  # 防止除以0

    # 計算新特徵（順序相反插入以保證最終順序正確）
    data.insert(0, 'DiabetesRiskIndex',
                (data['Glucose'] + data['BMI'] + data['Age']) / 3)
    data.insert(0, 'Glucose_BMI_product', data['Glucose'] * data['BMI'])
    data.insert(0, 'SkinThickness_BMI', data['SkinThickness'] * data['BMI'])
    data.insert(0, 'Age_Pregnancies', data['Age'] / (data['Pregnancies'] + 1))
    data.insert(0, 'Insulin_to_Glucose',
                data['Insulin'] / (data['Glucose'] + epsilon))
    data.insert(0, 'BMI_age_ratio', data['BMI'] / (data['Age'] + epsilon))

    # data.insert(0, 'N1', data['Age'] * data['Glucose'])  # 第0欄插入 C 欄
    # data.insert(0, 'N2', data['Age'] * data['Pregnancies'])  # 第1欄插入 D 欄
    # data.insert(0, 'N3', data['Glucose'] * data['BloodPressure'])  # 第2欄插入 E 欄

    return data
