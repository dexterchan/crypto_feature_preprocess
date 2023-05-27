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
    Feature_Output,
    create_feature_from_one_dim_data,
    create_feature_from_one_dim_data_v2,
    _initialize_price_feature_instance,
)
import pytest
from crypto_feature_preprocess.logging import get_logger
from functools import reduce

LOOK_BACK: int = 3
LOG_PRICE_LOOKBACK: int = 10
logger = get_logger(__name__)


@pytest.fixture()
def get_price_feature_spec() -> list[Feature_Definition]:
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


@pytest.fixture()
def get_vol_feature_spec() -> list[Feature_Definition]:
    feature_definitions = [
        Feature_Definition(
            meta={"name": Feature_Enum.LOG_PRICE},
            data=Log_Price_Feature_Interface(
                dimension=LOG_PRICE_LOOKBACK * 2, normalize_value=5
            ),
        )
    ]
    return feature_definitions


def test_teature_preparation_v2(
    get_test_decending_then_ascending_mkt_data, get_price_feature_spec
) -> None:
    candles = get_test_decending_then_ascending_mkt_data(dim=100)
    feature_schema_list = get_price_feature_spec

    one_vector_data = candles["close"]
    rsi_feature = _initialize_price_feature_instance(
        nt=feature_schema_list[1].data, price=one_vector_data
    )
    log_price_feature = _initialize_price_feature_instance(
        nt=feature_schema_list[0].data, price=one_vector_data
    )
    sma_cross_feature = _initialize_price_feature_instance(
        nt=feature_schema_list[2].data, price=one_vector_data
    )
    # Create reference feature for comparison
    rsi_feature_array = rsi_feature.output_feature_array(normalize=True)
    num_rsi_feature, rsi_dim = rsi_feature_array.shape

    log_price_feature_array = log_price_feature.output_feature_array(normalize=True)
    num_log_price_feature, log_dim = log_price_feature_array.shape

    sma_cross_feature_array = sma_cross_feature.output_feature_array(normalize=True)
    num_sma_cross_feature, sma_dim = sma_cross_feature_array.shape

    expected_feature_size: int = min(
        num_rsi_feature, num_log_price_feature, num_sma_cross_feature
    )
    expected_log_price_feature_array = log_price_feature_array[-expected_feature_size:]
    expected_rsi_feature_array = rsi_feature_array[-expected_feature_size:]
    expected_sma_cross_feature_array = sma_cross_feature_array[-expected_feature_size:]

    # Test our feature generation function
    feature_output: Feature_Output = create_feature_from_one_dim_data_v2(
        feature_schema_list=feature_schema_list,
        data_vector=one_vector_data,
    )

    assert feature_output.feature_data.shape == (
        expected_feature_size,
        rsi_dim + log_dim + sma_dim,
    ), "Feature dimension is not correct"

    # Check feature array here
    # Log Price feature, the difference between expected_log_price_feawture_array and feautre_array
    assert (
        np.abs(
            feature_output.feature_data[:, :log_dim] - expected_log_price_feature_array
        )
        < 0.1
    ).all()
    # RSI feature
    assert (
        np.abs(
            feature_output.feature_data[:, log_dim : log_dim + rsi_dim]
            - expected_rsi_feature_array
        )
        < 0.1
    ).all()
    # SMA feature
    assert (
        np.abs(
            feature_output.feature_data[:, log_dim + rsi_dim :]
            - expected_sma_cross_feature_array
        )
        < 0.1
    ).all()
    # Check meta data
    assert feature_output.metadata == feature_schema_list
    # Check timeindex
    assert (
        feature_output.time_index == one_vector_data.index[-expected_feature_size:]
    ).all()
    # Check price with timeindex
    assert (
        one_vector_data[one_vector_data.index == feature_output.time_index[0]]
        == one_vector_data[-expected_feature_size : -expected_feature_size + 1]
    ).all()
    pass


@pytest.mark.skip(reason="function deprecated")
def test_feature_preparation(
    get_test_decending_then_ascending_mkt_data, get_price_feature_spec
) -> None:
    candles = get_test_decending_then_ascending_mkt_data(dim=100)
    spec_list = get_price_feature_spec

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

    feature_array, feature_breakdown = create_feature_from_one_dim_data(
        price_vector=candles["close"], feature_list=spec_list
    )
    # Check feature array here
    num_features, dim = feature_array.shape
    assert num_features == expected_feature_size
    look_back_list: list[int] = [s.data.dimension for s in spec_list]
    assert dim == reduce((lambda x, y: x + y), look_back_list)

    expected_log_price_feature_array = log_price_feature_array[-expected_feature_size:]
    expected_rsi_feature_array = rsi_feature_array[-expected_feature_size:]
    expected_sma_cross_feature_array = sma_cross_feature_array[-expected_feature_size:]
    # Log Price feature, the difference between expected_log_price_feawture_array and feautre_array
    assert (
        np.abs(expected_log_price_feature_array - feature_array[:, : look_back_list[0]])
        < 0.1
    ).all()
    assert (
        np.abs(
            expected_log_price_feature_array - feature_array[:, : feature_breakdown[0]]
        )
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
    assert look_back_list[0] == feature_breakdown[0]
    assert look_back_list[1] + look_back_list[0] == feature_breakdown[1]

    assert (
        np.abs(
            expected_rsi_feature_array
            - feature_array[:, feature_breakdown[0] : feature_breakdown[1]]
        )
        < 0.1
    ).all()

    # check SMA Cross
    assert (
        np.abs(
            expected_sma_cross_feature_array
            - feature_array[:, look_back_list[0] + look_back_list[1] :]
        )
        < 0.1
    ).all()
    assert (
        np.abs(
            expected_sma_cross_feature_array - feature_array[:, feature_breakdown[1] :]
        )
        < 0.1
    ).all()
    pass


def test_feature_merging(
    get_test_decending_then_ascending_mkt_data,
    get_price_feature_spec,
    get_vol_feature_spec,
) -> None:
    candles = get_test_decending_then_ascending_mkt_data(dim=500)
    price_feature_schema_list = get_price_feature_spec
    volume_feature_schema_list = get_vol_feature_spec

    # Test our feature generation function
    price_feature_output: Feature_Output = create_feature_from_one_dim_data_v2(
        feature_schema_list=price_feature_schema_list,
        data_vector=candles["close"],
    )
    (
        price_feature_population_size,
        price_feature_length,
    ) = price_feature_output.feature_data.shape

    volume_feature_output: Feature_Output = create_feature_from_one_dim_data_v2(
        feature_schema_list=volume_feature_schema_list,
        data_vector=candles["volume"],
    )
    (
        volume_feature_population_size,
        volume_feature_length,
    ) = volume_feature_output.feature_data.shape

    new_feature_output: Feature_Output = Feature_Output.merge_feature_output_list(
        feature_output_list=[price_feature_output, volume_feature_output]
    )
    (
        new_feature_population_size,
        new_feature_length,
    ) = new_feature_output.feature_data.shape
    assert new_feature_population_size == min(
        price_feature_population_size, volume_feature_population_size
    ), "feature population size is not consistent"
    assert new_feature_length == (price_feature_length + volume_feature_length)

    assert len(new_feature_output.time_index) == min(
        price_feature_population_size, volume_feature_population_size
    ), "time index population size is not consistent"

    longest_time_index = (
        price_feature_output.time_index
        if price_feature_population_size > volume_feature_population_size
        else volume_feature_output.time_index
    )
    assert (
        new_feature_output.time_index
        == longest_time_index[-new_feature_population_size:]
    ).all(), "time index content is not consistent"

    # Check the feature content
    # Check the price feature consistent with the original feature
    assert (
        new_feature_output.feature_data[:, :price_feature_length]
        == price_feature_output.feature_data[-new_feature_population_size:, :]
    ).all(), "price feature content is not consistent"
    # Check the volume feature consistent with the original feature
    assert (
        new_feature_output.feature_data[:, price_feature_length:]
        == volume_feature_output.feature_data[-new_feature_population_size:, :]
    ).all(), "volume feature content is not consistent"
    pass
