"""
Copyright 2023

Author: Lukas Schweizer <schweizer.lukas@web.de>
"""

#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import numpy as np
import pandas as pd

import torch


def preprocess_data(
    data,
    nan_values=True,
    one_hot_encoding=False,
    normalization=True,
    categorical_indices=None,
):
    """
    This method preprocesses data regarding missing values, categorical features
    and data normalization (for the kNN Model)
    :param data: Data to preprocess
    :param nan_values: Preprocesses nan values if True
    :param one_hot_encoding: Whether use OHE for categoricals
    :param normalization: Normalizes data if True
    :param categorical_indices: Categorical columns of data
    :return: Preprocessed version of the data
    """
    data = data.numpy() if torch.is_tensor(data) else data
    data = data.astype(np.float32)
    data = pd.DataFrame(data).reset_index().drop("index", axis=1)

    if categorical_indices is None:
        categorical_indices = []
    preprocessed_data = data
    # NaN values (replace NaN with zeros)
    if nan_values:
        preprocessed_data = preprocessed_data.fillna(0)
    # Categorical Features (One Hot Encoding)
    if one_hot_encoding:
        # Setting dtypes of categorical data to 'category'
        for idx in categorical_indices:
            preprocessed_data[preprocessed_data.columns[idx]] = preprocessed_data[
                preprocessed_data.columns[idx]
            ].astype("category")
        categorical_columns = list(
            preprocessed_data.select_dtypes(include=["category"]).columns
        )
        preprocessed_data = pd.get_dummies(
            preprocessed_data, columns=categorical_columns
        )
    # Data normalization from R -> [0, 1]
    if normalization:
        if one_hot_encoding:
            numerical_columns = list(
                preprocessed_data.select_dtypes(exclude=["category"]).columns
            )
            preprocessed_data[numerical_columns] = preprocessed_data[
                numerical_columns
            ].apply(
                lambda x: (x - x.min()) / (x.max() - x.min())
                if x.max() != x.min()
                else x
            )
        else:
            preprocessed_data = preprocessed_data.apply(
                lambda x: (x - x.min()) / (x.max() - x.min())
                if x.max() != x.min()
                else x
            )
    return preprocessed_data
