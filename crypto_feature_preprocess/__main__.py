# Main file to generate training and evaluation raw data

import argparse
from argparse import RawTextHelpFormatter
import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta

from crypto_feature_preprocess.logging import get_logger
from crypto_feature_preprocess.port.training_data_parquet import (
    prepare_training_data_and_eval_from_parquet,
    derive_min_candle_population_in_episode,
)
from datetime import datetime, timedelta

logger = get_logger(__name__)

if __name__ == "__main__":
    # Collect arguments
    # exchange
    # symbol
    # input data directory
    # output data directory
    # start date YYYYMMDD
    # time windows (days)
    # data length (days)
    # data step (days)
    # split ratio
    # candle size in minutes

    parser = argparse.ArgumentParser(
        description="Generate training and evaluation raw data",
        formatter_class=RawTextHelpFormatter,
    )
    # parser accept help here

    parser.add_argument(
        "--exchange",
        type=str,
        required=True,
        help="Exchange name",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Symbol name",
    )
    parser.add_argument(
        "--input_data_dir",
        type=str,
        required=True,
        help="Input data directory",
    )
    parser.add_argument(
        "--output_data_dir",
        type=str,
        required=True,
        help="Output data directory",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date in YYYYMMDD format",
    )
    parser.add_argument(
        "--time_windows",
        type=int,
        required=True,
        help="Time windows in days",
        default=30,
    )
    parser.add_argument(
        "--data_length", type=int, required=True, help="Data length in days", default=3
    )
    parser.add_argument(
        "--data_step", type=int, required=True, help="Data step in days", default=1
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        required=True,
        help="Split ratio of training data in the data set e.g. 0.8",
        default=0.8,
    )
    parser.add_argument(
        "--candle_size",
        type=int,
        required=True,
        help="Candle size in minutes",
        default=15,
    )

    args = parser.parse_args()

    # Parse arguments
    exchange: str = args.exchange
    symbol: str = args.symbol
    input_data_dir: str = args.input_data_dir
    output_data_dir: str = args.output_data_dir
    start_date_yyyymmdd: str = args.start_date
    time_windows: int = args.time_windows
    data_length_days: int = args.data_length
    data_step: int = args.data_step
    split_ratio: float = args.split_ratio
    candle_size_minutes: int = args.candle_size

    # Convert start_date_yyyymmdd to datetime
    start_date: datetime = datetime.strptime(start_date_yyyymmdd, "%Y%m%d")
    # calculate the end date
    end_date: datetime = start_date + timedelta(days=time_windows)

    min_candle_population: int = derive_min_candle_population_in_episode(
        candle_size_minutes=candle_size_minutes,
        data_length_days=data_length_days,
        data_presence_ratio=0.9,
    )

    # Generate training and evaluation data
    (
        num_training_data_row,
        num_eval_data_row,
    ) = prepare_training_data_and_eval_from_parquet(
        exchange=exchange,
        symbol=symbol,
        data_type="parquet",
        data_directory=input_data_dir,
        start_date=start_date,
        end_date=end_date,
        data_length=timedelta(days=data_length_days),
        data_step=timedelta(days=data_step),
        split_ratio=split_ratio,
        output_folder=output_data_dir,
        candle_size=f"{candle_size_minutes}Min",
        min_candle_population=min_candle_population,
    )

    logger.info(
        f"Generated training data: {num_training_data_row} rows, evaluation data: {num_eval_data_row} rows"
    )
