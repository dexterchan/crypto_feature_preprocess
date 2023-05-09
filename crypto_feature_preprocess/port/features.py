from ..domains.features_gen import (
    Log_Price_Feature,
    RSI_Feature,
    SMA_Cross_Feature,
    Feature,
)
from .interfaces import (
    Feature_Definition,
    Feature_Enum,
    SMA_Cross_Feature_Interface,
    Log_Price_Feature_Interface,
    RSI_Feature_Interface,
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


def create_feature_from_close_price(
    ohlcv_candles: pd.DataFrame, feature_pools: list[Feature_Definition]
) -> np.ndarray:
    """Create feature vectors from close price

    Args:
        close_price (pd.DataFrame): OHLCV candles in pandas dataframe
        feature_pools (list[Feature_Definition]): feature to aggregrate

    Returns:
        np.ndarray: array of features
    """

    close_price = ohlcv_candles["close"]

    feature_set: list[np.ndarray] = []
    for f_def in feature_pools:
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
    return feature_set
