# testing for preprocess.domain.features.SMA_Cross_Feature
from crypto_feature_preprocess.domains.features_gen import SMA_Cross_Feature
from crypto_feature_preprocess.domains.indicators import calculate_simple_moving_average
from crypto_feature_preprocess.port.interfaces import SMA_Cross_Feature_Interface
import numpy as np
import pytest
import logging

logger = logging.getLogger(__name__)

LOOK_BACK: int = 3


def test_sma_calcuation(get_test_ascending_mkt_data) -> None:
    length = 100
    close_price = get_test_ascending_mkt_data(dim=length)["close"]
    sma_5 = calculate_simple_moving_average(df_price=close_price, window=5)
    sma_5.dropna(inplace=True)
    assert len(sma_5) == len(close_price) - 5 + 1

    sma_10 = calculate_simple_moving_average(df_price=close_price, window=10)
    sma_10.dropna(inplace=True)
    assert len(sma_10) == len(close_price) - 10 + 1


def test_sma_cross_over(get_test_decending_then_ascending_mkt_data) -> None:
    length = 16
    close_price = get_test_decending_then_ascending_mkt_data(dim=length)["close"]

    sma_cross = SMA_Cross_Feature(
        df_price=close_price, sma_window_1=5, sma_window_2=10, dimension=LOOK_BACK
    )

    cross_over = sma_cross._calculate()

    assert len(cross_over) == len(close_price) - sma_cross.invalid_data_length

    sma_cross_indices = np.where(cross_over == True)[0] + sma_cross.invalid_data_length
    assert len(sma_cross_indices) == 1

    #logger.info(f"cross_over: {cross_over}")

    sma_cross_features = sma_cross.output_feature_array()
    num_features, dim = sma_cross.shape
    assert len(sma_cross_features) == num_features
    assert dim == LOOK_BACK

    ref_data = [
        [
            0,
            0,
            0,
        ],
        [
            0,
            0,
            0,
        ],
        [
            1,
            0,
            0,
        ],
        [
            0,
            1,
            0,
        ],
        [
            0,
            0,
            1,
        ],
    ]

    #logger.info(sma_cross_features)
    assert (ref_data == sma_cross_features).all()

def test_initialization_from_port(get_test_decending_then_ascending_mkt_data) -> None:
    close_price = get_test_decending_then_ascending_mkt_data(dim=1000)["close"]
    sma_cross_interface = SMA_Cross_Feature_Interface(
        sma_window_1=20,
        sma_window_2=100,
        dimension=3
    )

    sma_feature = SMA_Cross_Feature(**(sma_cross_interface._asdict()), df_price=close_price)
    sma_feature_array = sma_feature.output_feature_array()
    num_features, dim = sma_feature.shape
    assert len(sma_feature_array) == num_features
    assert dim == sma_feature_array.shape[1]
    
