from sklearn.model_selection import train_test_split
from utils.config import SEED


def split_data(data, train_size=0.8, valid_size=0.1, test_size=0.1):
    """
    Split the data into train, validation, and test sets.
    :param data: DataFrame containing the data
    :param valid_size: Proportion of the data to use for validation
    :param test_size: Proportion of the data to use for testing
    :return: X_train, X_valid, X_test, y_train, y_valid, y_test
    """

    if train_size + valid_size + test_size != 1.0:
        raise ValueError("train_size + valid_size + test_size must equal 1.0")

    df_tmp, df_test = train_test_split(data,
                                       test_size=test_size,
                                       random_state=SEED)
    df_train, df_valid = train_test_split(df_tmp,
                                          test_size=valid_size / train_size,
                                          random_state=SEED)
    return df_train, df_valid, df_test


def get_features_and_target(data):
    """
    Split the data into features and target variable.
    :param data: DataFrame containing the data
    :return: X, y
    """

    X = data.iloc[:, :-1].values.astype('float32')
    y = data.iloc[:, -1].values.astype('int64')
    return X, y
