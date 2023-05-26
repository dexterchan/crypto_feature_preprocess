from ..domains.features_gen import (
    Log_Price_Feature,
    RSI_Feature,
    SMA_Cross_Feature,
    Feature,
)
from .interfaces import (
    Feature_Definition,
    SMA_Cross_Feature_Interface,
    Log_Price_Feature_Interface,
    RSI_Feature_Interface,
    Feature_Output,
)

import pandas as pd
import numpy as np
from typing import NamedTuple
from crypto_feature_preprocess.logging import get_logger

logger = get_logger(__name__)


def _initialize_price_feature_instance(nt: NamedTuple, price: pd.Series) -> Feature:
    """Initialize feature instance from namedtuple"""
    if isinstance(nt, Log_Price_Feature_Interface):
        return Log_Price_Feature(**(nt._asdict()), df_price=price)
    elif isinstance(nt, SMA_Cross_Feature_Interface):
        return SMA_Cross_Feature(**(nt._asdict()), df_price=price)
    elif isinstance(nt, RSI_Feature_Interface):
        return RSI_Feature(**(nt._asdict()), df_price=price)
    else:
        raise NotImplementedError(f"Not supporting this config: {nt}")


def trim_feature_to_same_length_and_group(
    feature_list: list[np.ndarray], time_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Trim all feature to the same length and group them together

    Args:
        feature_list (list[np.ndarray]): list of feature

    Returns:
        tuple (np.ndarray, np.ndarray): feature with same length, time index
    """
    # Find the shortest feature length in feature_set
    shortest_feature_length = min([len(f) for f in feature_list])
    # Trim all feature to the same length
    feature_list = [
        f[-shortest_feature_length:].reshape(shortest_feature_length, -1)
        for f in feature_list
    ]
    feature_set = np.concatenate(feature_list, axis=1)
    new_time_index = time_index[-shortest_feature_length:]
    return feature_set, new_time_index


def create_feature_from_one_dim_data_v2(
    data_vector: pd.Series, feature_schema_list: list[Feature_Definition]
) -> Feature_Output:
    """Create feature vectors from 1 dimension data vector

    Args:
        data_vector (pd.Series): One dimension data vector e.g. close price, volume
        feature_schema_list (list[Feature_Definition]): feature to aggregate

    Returns:
        Feature_Output: Feature output data
    """
    time_index = data_vector.index

    feature_output_list: list[np.ndarray] = []
    for f_def in feature_schema_list:
        # initialize feature instance from namedtuple
        feature = _initialize_price_feature_instance(nt=f_def.data, price=data_vector)
        feature_array = feature.output_feature_array(normalize=True)
        feature_output_list.append(feature_array)

    new_feature_data, new_time_index = trim_feature_to_same_length_and_group(
        feature_list=feature_output_list, time_index=time_index
    )

    return Feature_Output(
        metadata=feature_schema_list,
        time_index=new_time_index,
        feature_data=new_feature_data,
    )


def create_feature_from_one_dim_data(
    price_vector: pd.Series, feature_list: list[Feature_Definition]
) -> tuple[np.ndarray, np.array]:
    """Create feature vectors from close price

    Args:
        price_vector (pd.Series): 1 dimension price vector
        feature_pools (list[Feature_Definition]): feature to aggregrate

    Returns:
        tuple[] : [N X (accm of feature dimension)] matrix of features, feature breakdown in array

    """

    close_price = price_vector

    feature_set: list[np.ndarray] = []
    for f_def in feature_list:
        # initialize feature instance from namedtuple
        feature = _initialize_price_feature_instance(nt=f_def.data, price=close_price)
        feature_array = feature.output_feature_array(normalize=True)
        feature_set.append(feature_array)
    # Find the shortest feature length in feature_set
    shortest_feature_length = min([len(f) for f in feature_set])
    # Trim all feature to the same length
    feature_set = [
        f[-shortest_feature_length:].reshape(shortest_feature_length, -1)
        for f in feature_set
    ]
    # Concatenate all features
    feature_set = np.concatenate(feature_set, axis=1)

    features_spec_output_list: list[int] = [
        f_def.data.dimension for f_def in feature_list
    ]
    accm = np.array(features_spec_output_list)
    feature_breakdown = np.add.accumulate(accm)

    return feature_set, feature_breakdown
