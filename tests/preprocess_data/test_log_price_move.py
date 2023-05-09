# testing for preprocess.domain.features.log_price_move.LogPriceMove
from crypto_feature_preprocess.domains.features_gen import Log_Price_Feature, Feature
import math
import numpy as np
import logging
import pytest

logger = logging.getLogger(__name__)

TOLERANCE: float = 0.0001

LOOK_BACK: int = 3


class DummyFeature(Feature):
    def __init__(self) -> None:
        pass

    def output_feature_array(self) -> np.ndarray:
        return []


def test_look_back_feature_copy() -> None:
    raw_data_length = 10
    dim: int = LOOK_BACK
    raw_data = np.arange(raw_data_length)
    dummy_feature = DummyFeature()
    features = dummy_feature.create_feature_from_raw_data_array(
        raw_data_array=raw_data, look_back=dim
    )

    num_features = dummy_feature.feature_length(
        raw_data_length=len(raw_data), look_back=dim
    )
    assert num_features == raw_data_length - LOOK_BACK + 1

    ref_data: np.ndarray = np.array(
        [
            [
                2,
                1,
                0,
            ],
            [
                3,
                2,
                1,
            ],
            [
                4,
                3,
                2,
            ],
            [
                5,
                4,
                3,
            ],
            [
                6,
                5,
                4,
            ],
            [
                7,
                6,
                5,
            ],
            [
                8,
                7,
                6,
            ],
            [
                9,
                8,
                7,
            ],
        ]
    )
    # logger.info(raw_data)
    # logger.info(f"features: {features}")
    assert (features == ref_data).all()


def test_output_feature_array(get_test_ascending_mkt_data) -> None:
    df = get_test_ascending_mkt_data(dim=10)
    dim: int = LOOK_BACK
    log_price_move = Log_Price_Feature(df["close"], dim)
    price_move = log_price_move._calculate()
    feature_array = log_price_move.output_feature_array()

    # logger.info(price_move)
    # logger.info(feature_array)

    assert feature_array.shape == (len(df) - LOOK_BACK, dim)
    # Check if the first element is correct and within the tolerance TOLERANCE
    assert len(price_move) == len(df) - 1
    assert (price_move[0] - math.log(df["close"][1] / df["close"][0])) < TOLERANCE

    # Check if feature_array[0] is the same as price_move[0:dim] (difference is smaller than TOLERANCE)
    assert ((feature_array[0] - price_move[0:dim].values[::-1]) < TOLERANCE).all()
    # Enumerate each vector of feature_array and check if it is the same as price_move[i:i+dim]
    for i, v in enumerate(feature_array):
        assert len(v) == dim
        # Reverse price move order
        ref = price_move[i : i + dim].values[::-1]
        assert ((v - ref) < TOLERANCE).all()
        if i == 0:
            break
    pass
