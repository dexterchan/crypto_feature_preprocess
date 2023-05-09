from __future__ import annotations
from crypto_feature_preprocess.port.interfaces import (
    Feature_Definition,
    RSI_Feature_Interface,
    Log_Price_Feature_Interface,
    SMA_Cross_Feature_Interface,
    Feature_Enum,
)
import numpy as np
from crypto_feature_preprocess.port.features import (
    create_feature_from_close_price,
    _initialize_price_feature_instance,
)
import pytest
from crypto_feature_preprocess.logging import get_logger
from functools import reduce

LOOK_BACK: int = 3
LOG_PRICE_LOOKBACK: int = 10
logger = get_logger(__name__)


@pytest.fixture()
def get_feature_spec() -> list[Feature_Definition]:
    feature_definitions = [
        Feature_Definition(
            meta={"name": Feature_Enum.LOG_PRICE},
            data=Log_Price_Feature_Interface(dimension=LOG_PRICE_LOOKBACK),
        ),
        Feature_Definition(
            meta={"name": Feature_Enum.RSI},
            data=RSI_Feature_Interface(rsi_window=14, dimension=LOOK_BACK),
        ),
        Feature_Definition(
            meta={"name": Feature_Enum.SMA_CROSS},
            data=SMA_Cross_Feature_Interface(
                sma_window_1=20, sma_window_2=50, dimension=LOOK_BACK
            ),
        ),
    ]
    return feature_definitions


def test_feature_preparation(
    get_test_decending_then_ascending_mkt_data, get_feature_spec
) -> None:
    candles = get_test_decending_then_ascending_mkt_data(dim=100)
    spec_list = get_feature_spec

    rsi_feature = _initialize_price_feature_instance(
        nt=spec_list[1].data, price=candles["close"]
    )
    log_price_feature = _initialize_price_feature_instance(
        nt=spec_list[0].data, price=candles["close"]
    )
    sma_cross_feature = _initialize_price_feature_instance(
        nt=spec_list[2].data, price=candles["close"]
    )
    rsi_feature_array = rsi_feature.output_feature_array(normalize=True)
    num_rsi_feature, dim = rsi_feature_array.shape

    log_price_feature_array = log_price_feature.output_feature_array(normalize=True)
    num_log_price_feature, dim = log_price_feature_array.shape

    sma_cross_feature_array = sma_cross_feature.output_feature_array(normalize=True)
    num_sma_cross_feature, dim = sma_cross_feature_array.shape

    expected_feature_size = min(
        num_log_price_feature, num_sma_cross_feature, num_rsi_feature
    )

    logger.debug(f"RSI feature size: {num_rsi_feature}")
    logger.debug(f"Log price feature size: {num_log_price_feature}")
    logger.debug(f"SMA cross feature size: {num_sma_cross_feature}")
    logger.debug(f"Expected feature size: {expected_feature_size}")

    feature_array = create_feature_from_close_price(
        ohlcv_candles=candles, feature_pools=spec_list
    )
    # Check feature array here
    num_features, dim = feature_array.shape
    assert num_features == expected_feature_size
    look_back_list: list[int] = [s.data.dimension for s in spec_list]
    assert dim == reduce((lambda x, y: x + y), look_back_list)

    expected_log_price_feature_array = log_price_feature_array[-expected_feature_size:]
    expected_rsi_feature_array = rsi_feature_array[-expected_feature_size:]
    expected_sma_cross_feature_array = sma_cross_feature_array[-expected_feature_size:]
    # assert (
    #     expected_log_price_feature_array == feature_array[:, : look_back_list[0]]
    # ).all()
    assert (
        np.abs(expected_log_price_feature_array - feature_array[:, : look_back_list[0]])
        < 0.1
    ).all()
    # Check RSI feature, the difference between expected_rsi_feature_array and feature_array[:, look_back_list[0] : look_back_list[0] + look_back_list[1]]  is less than 0.1
    assert (
        np.abs(
            expected_rsi_feature_array
            - feature_array[
                :, look_back_list[0] : look_back_list[0] + look_back_list[1]
            ]
        )
        < 0.1
    ).all()

    # assert (
    #     expected_rsi_feature_array
    #     == feature_array[:, look_back_list[0] : look_back_list[0] + look_back_list[1]]
    # ).all()
    assert (
        np.abs(
            expected_sma_cross_feature_array
            - feature_array[:, look_back_list[0] + look_back_list[1] :]
        )
        < 0.1
    ).all()
    # assert (
    #     expected_sma_cross_feature_array
    #     == feature_array[:, look_back_list[0] + look_back_list[1] :]
    # ).all()
    pass
