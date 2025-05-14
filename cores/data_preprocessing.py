import numpy as np
from utils.config import SEED
from utils.logger_util import CustomLogger as logger
import pandas as pd

from models.MICE import ScaledRegressor
from sklearn.linear_model import LinearRegression


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


def impute_missing_values_with_MICE(data,
                                    target_cols,
                                    ignore_cols=None,
                                    estimator=None,
                                    max_iter=10,
                                    seed=0):
    """
    Impute missing values only for target columns using MICE, but with all other columns as predictors.

    Args:
        data (pd.DataFrame): Input dataset.
        target_cols (list): Columns to be imputed (i.e., only these will be updated).
        ignore_cols (list): Columns to exclude from predictors (e.g., labels).
        estimator (sklearn estimator): Regression model used by IterativeImputer.
        max_iter (int): Number of MICE iterations.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Data with target columns imputed.
    """
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    logger.info("Imputing missing values with MICE...")

    if estimator is None:
        estimator = ScaledRegressor()

    data_copy = data.copy()

    # Define predictor columns (exclude ignore_cols)
    feature_cols = data.columns.difference(ignore_cols if ignore_cols else [])

    # Replace invalid 0s in target_cols with NaN
    data_copy[target_cols] = data_copy[target_cols].replace(0, np.nan)

    # Run imputer on full predictor set
    imputer = IterativeImputer(estimator=estimator,
                               max_iter=max_iter,
                               initial_strategy='median',
                               random_state=seed)
    imputed_all = imputer.fit_transform(data_copy[feature_cols])

    # Convert back to DataFrame
    imputed_df = pd.DataFrame(imputed_all, columns=feature_cols)

    # Replace only the target columns in original data
    data_copy[target_cols] = imputed_df[target_cols]

    return data_copy


def impute_missing_values_with_MICE_wo_data_leak(train_data,
                                                 val_data,
                                                 test_data,
                                                 target_cols,
                                                 ignore_cols=None,
                                                 max_iter=10,
                                                 seed=0):
    """
    Impute missing values in the dataset by replacing zeros with NaN,
    removing outliers temporarily, and filling missing values with the median.
    
    Specifically designed to avoid data leakage between train, validation, and test sets.

    Args:
        train_data (pd.DataFrame): Training dataset.
        val_data (pd.DataFrame): Validation dataset.
        test_data (pd.DataFrame): Testing dataset.
    Returns:
        tuple: Tuple containing the training, validation, and testing datasets with imputed values.
    """

    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    logger.info("Imputing missing values with MICE...")
    estimator = ScaledRegressor()

    # Define predictor columns (exclude ignore_cols)
    feature_cols = train_data.columns.difference(
        ignore_cols if ignore_cols else [])

    # Replace invalid 0s in target_cols with NaN
    train_data[target_cols] = train_data[target_cols].replace(0, np.nan)
    val_data[target_cols] = val_data[target_cols].replace(0, np.nan)
    test_data[target_cols] = test_data[target_cols].replace(0, np.nan)
    # Run imputer on full predictor set
    imputer = IterativeImputer(estimator=estimator,
                               max_iter=max_iter,
                               initial_strategy='median',
                               random_state=seed)
    imputer.fit(train_data[feature_cols])
    imputed_train = imputer.transform(train_data[feature_cols])
    imputed_val = imputer.transform(val_data[feature_cols])
    imputed_test = imputer.transform(test_data[feature_cols])

    # Convert back to DataFrame
    imputed_train = pd.DataFrame(imputed_train,
                                 columns=feature_cols,
                                 index=train_data.index)
    imputed_val = pd.DataFrame(imputed_val,
                               columns=feature_cols,
                               index=val_data.index)
    imputed_test = pd.DataFrame(imputed_test,
                                columns=feature_cols,
                                index=test_data.index)
    # Replace only the target columns in original data
    train_data[target_cols] = imputed_train[target_cols]
    val_data[target_cols] = imputed_val[target_cols]
    test_data[target_cols] = imputed_test[target_cols]

    return train_data, val_data, test_data


def impute_missing_values_wo_data_leak(train_data, val_data, test_data):
    """
    Impute missing values in the dataset by replacing zeros with NaN,
    removing outliers temporarily, and filling missing values with the median.
    
    Specifically designed to avoid data leakage between train, validation, and test sets.

    Args:
        train_data (pd.DataFrame): Training dataset.
        val_data (pd.DataFrame): Validation dataset.
        test_data (pd.DataFrame): Testing dataset.
    Returns:
        tuple: Tuple containing the training, validation, and testing datasets with imputed values.
    """

    # All of attributes: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome]
    fix_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for col in fix_cols:
        # Replace zeros with NaN
        train_data[col] = train_data[col].replace(0, np.nan)
        val_data[col] = val_data[col].replace(0, np.nan)
        test_data[col] = test_data[col].replace(0, np.nan)

        # Remove outliers temporarily and calculate median
        clean_series = remove_outliers_iqr(train_data[col].dropna())

        skew = clean_series.skew()
        if abs(skew) >= 0.5:
            # If skewness is high, use the median of the original series
            fill_value = train_data[col].median()
        else:
            # If skewness is low, use the mean of the cleaned series
            fill_value = clean_series.mean()

        # Fill missing values with the calculated median
        train_data[col] = train_data[col].fillna(fill_value)
        val_data[col] = val_data[col].fillna(fill_value)
        test_data[col] = test_data[col].fillna(fill_value)

    return train_data, val_data, test_data


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
