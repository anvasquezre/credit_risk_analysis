from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.

    object_cols = list(working_train_df.select_dtypes(include="object").columns)
    low_cardinality_cols = [
        col
        for col in working_train_df[object_cols].columns
        if working_train_df[col].nunique() <= 2
    ]
    high_cardinality_cols = [
        col
        for col in working_train_df[object_cols].columns
        if working_train_df[col].nunique() > 2
    ]

    OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    OR_encoder = OrdinalEncoder(handle_unknown="error")

    OH_encoder.fit(working_test_df[high_cardinality_cols])
    OR_encoder.fit(working_test_df[low_cardinality_cols])

    dfs = {
        x: y
        for (x, y) in enumerate([working_train_df, working_val_df, working_test_df])
    }

    dfs_hc_encoded = {}
    dfs_lc_encoded = {}

    for index, value in dfs.items():
        # OH encoding of > 2 cardinality cols
        dfs_hc_encoded[index] = pd.DataFrame(
            OH_encoder.transform(value[high_cardinality_cols])
        )
        # OH Removes indexes, adding them again
        dfs_hc_encoded[index].index = value[high_cardinality_cols].index

        # Ordinal encoding of <= 2 cardinality cols
        dfs_lc_encoded[index] = pd.DataFrame(
            OR_encoder.transform(value[low_cardinality_cols])
        )
        # OR Removes indexes, adding them again
        dfs_lc_encoded[index].index = value[low_cardinality_cols].index
        # Changing column names to avoid miss indexing when joining
        initial_col = dfs_hc_encoded[index].shape[1]
        # First column of lc should be the next number of hc
        final_col = dfs_hc_encoded[index].shape[1] + dfs_lc_encoded[index].shape[1]
        dfs_lc_encoded[index].columns = [
            col_num for col_num in range(initial_col, final_col)
        ]
        # Dropping encoded cols from original data and merge with encoded
        dfs[index] = (
            value.drop(columns=[*high_cardinality_cols, *low_cardinality_cols])
            .join(dfs_hc_encoded[index])
            .join(dfs_lc_encoded[index])
        )
        # Setting column names as str to avoid problems with simple imputer
        dfs[index].columns = dfs[index].columns.astype(str)

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.

    imputer = SimpleImputer(strategy="median")
    imputer.fit(dfs[0])

    for index, value in dfs.items():
        dfs[index] = imputer.transform(value)

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.

    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(dfs[0])

    for index, value in dfs.items():
        dfs[index] = minmax_scaler.transform(value)

    return dfs[0], dfs[1], dfs[2]
