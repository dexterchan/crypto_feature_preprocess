from __future__ import annotations

from crypto_feature_preprocess.domains.features_gen import RSI_Feature
from crypto_feature_preprocess.domains.indicators import calculate_rsi

import numpy as np
import pytest
import logging
import pandas as pd

logger = logging.getLogger(__name__)


TOLERANCE: float = 0.01
LOOK_BACK: int = 3


def test_rsi_indicator(get_test_decending_then_ascending_mkt_data) -> None:
    dim: int = 100
    rsi_window: int = 14
    mktdata_close = get_test_decending_then_ascending_mkt_data(dim=dim)["close"]

    raw_rsi = calculate_rsi(df_price=mktdata_close, window=rsi_window)
    raw_rsi.dropna(inplace=True)
    assert len(raw_rsi) == len(mktdata_close) - rsi_window
    pass


def test_rsi_feature(get_test_decending_then_ascending_mkt_data) -> None:
    dim: int = 20
    rsi_window: int = 14
    mktdata_close = get_test_decending_then_ascending_mkt_data(dim=dim)["close"]

    rsi_feature = RSI_Feature(
        df_price=mktdata_close, dimension=LOOK_BACK, rsi_window=rsi_window
    )

    # raw_rsi = rsi_feature._calculate()
    feature_array = rsi_feature.output_feature_array()

    assert len(feature_array) == len(mktdata_close) - LOOK_BACK - rsi_window + 1
    num_features, feature_length = rsi_feature.shape
    assert feature_length == LOOK_BACK
    assert num_features == len(mktdata_close) - LOOK_BACK - rsi_window

    # logger.info(f"raw_rsi: {raw_rsi}")
    # logger.info(f"feature_array: {feature_array}")

    ref_data = [
        [51.68643849, 46.14758625, 39.73172763],
        [56.50421313, 51.68643849, 46.14758625],
        [60.72223545, 56.50421313, 51.68643849],
        [64.43632986, 60.72223545, 56.50421313],
    ]
    assert np.allclose(feature_array, ref_data, atol=TOLERANCE)

    normalized_feature_array = rsi_feature.output_feature_array(normalize=True)

    assert (normalized_feature_array < 1).all()
    assert (normalized_feature_array > -1).all()
    assert (normalized_feature_array < 0).any()
    assert (normalized_feature_array > 0).any()
    # logger.info(f"normalized_feature_array: {normalized_feature_array}")

    pass
