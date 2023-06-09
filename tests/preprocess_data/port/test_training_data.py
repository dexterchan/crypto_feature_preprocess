from __future__ import annotations

from crypto_feature_preprocess.logging import get_logger
from crypto_feature_preprocess.port.training_data_parquet import (
    prepare_training_data_and_eval_from_parquet,
    derive_min_candle_population_in_episode,
)
from datetime import datetime, timedelta

import pytest
import os

logger = get_logger(__name__)

test_symbol = "ETHUSD"
test_exchange = "kraken"
test_data_dir = os.environ.get("DATA_DIR", "notebooks/data")
test_output_dir = os.environ.get("OUTPUT_DIR", "data/output")


def test_prepare_training_data_and_eval_from_parquet() -> None:
    data_length_days = 3
    data_step_days = 1
    time_windows_days = 30
    start_date = datetime(2020, 1, 1)
    candle_size_min: int = 15

    end_date = start_date + timedelta(days=time_windows_days)
    data_step: timedelta = timedelta(days=data_step_days)
    data_length = timedelta(days=data_length_days)
    split_ratio = 0.8

    min_num_training_data_row = (
        4 * 24 * (time_windows_days - data_length_days + 1) * split_ratio * 0.9
    )
    min_num_eval_data_row = (
        4 * 24 * (time_windows_days - data_length_days + 1) * (1 - split_ratio) * 0.9
    )

    # output_folder should be data/training/YYYYMMDD
    output_folder = f"{test_output_dir}/training/{test_exchange}/{test_symbol}/{datetime.now().strftime('%Y%m%d')}"

    min_candle_population: int = derive_min_candle_population_in_episode(
        candle_size_minutes=candle_size_min,
        data_length_days=data_length_days,
        data_presence_ratio=0.8,
    )
    (
        num_training_data_row,
        num_eval_data_row,
    ) = prepare_training_data_and_eval_from_parquet(
        exchange=test_exchange,
        symbol=test_symbol,
        data_type="parquet",
        data_directory=test_data_dir,
        start_date=start_date,
        end_date=end_date,
        data_length=data_length,
        data_step=data_step,
        split_ratio=split_ratio,
        output_folder=output_folder,
        candle_size=f"{candle_size_min}Min",
        min_candle_population=min_candle_population,
    )
    assert num_training_data_row >= min_num_training_data_row
    assert num_eval_data_row >= min_num_eval_data_row
