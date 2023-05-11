from __future__ import annotations

from crypto_feature_preprocess.logging import get_logger
from crypto_feature_preprocess.port.training_data_parquet import (
    prepare_training_data_and_eval_from_parquet,
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
    start_date = datetime(2020, 1, 1)
    end_date = start_date + timedelta(days=365 * 3)
    data_length = timedelta(days=data_length_days)
    split_ratio = 0.8

    # output_folder should be data/training/YYYYMMDD
    output_folder = f"{test_output_dir}/training/{test_exchange}/{test_symbol}/{datetime.now().strftime('%Y%m%d')}"

    prepare_training_data_and_eval_from_parquet(
        exchange=test_exchange,
        symbol=test_symbol,
        data_type="parquet",
        data_directory=test_data_dir,
        start_date=start_date,
        end_date=end_date,
        data_length=data_length,
        split_ratio=split_ratio,
        output_folder=output_folder,
        candle_size="15Min",
        min_candle_population=int(4 * 24 * data_length_days * 0.8),
    )
