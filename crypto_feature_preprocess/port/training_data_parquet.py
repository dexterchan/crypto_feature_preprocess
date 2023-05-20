from cryptomarketdata.port.db_client import get_data_db_client, Database_Type
from cryptomarketdata.utility import resample_timeframe
from datetime import datetime, timedelta
from ..domains.training_data import splitting_training_and_eval_time_range
import pandas as pd
from ..adapter.TrainingDataStorage import TrainingDataStorage
import os
from ..logging import get_logger
from .interfaces import Training_Eval_Enum

logger = get_logger(__name__)


def derive_min_candle_population_in_episode(
    candle_size_minutes: int, data_length_days: int, data_presence_ratio: float = 0.8
) -> int:
    min_candle_population: int = int(
        timedelta(days=1)
        / timedelta(minutes=candle_size_minutes)
        * data_length_days
        * data_presence_ratio
    )
    return min_candle_population


def prepare_training_data_and_eval_from_parquet(
    exchange: str,
    symbol: str,
    data_directory: str,
    start_date: datetime,
    end_date: datetime,
    data_length: timedelta,
    data_step: timedelta,
    split_ratio: float,
    output_folder: str,
    candle_size: str,
    min_candle_population: int,
    data_type: str = "PARQUET",
) -> tuple[int, int]:
    """Prepare training data and eval data from parquet file
       It outputs OHLVC data in parquet file format with schema:
        open: double
        high: double
        low: double
        close: double
        volume: double
        scenario: int64
        __index_level_0__: timestamp[us]
        -- schema metadata --
        pandas: '{"index_columns": ["__index_level_0__"], "column_indexes": [{"na' + 1003

    Args:

        exchange (str): exchange name
        symbol (str): symbol name
        data_directory (str): data directory
        start_date (datetime): start date
        end_date (datetime): end date
        data_length (timedelta): data length
        data_step (timedelta): data step
        split_ratio (float): split ratio
        output_folder (str): output folder
        candle_size (str): candle size e.g. 15Min, 1H, D
        min_candle_population (int): min candle population
        data_type (Database_Type): data type

    Returns:
        tuple[int, int]: written training data(num of rows) and eval data(num of rows)
    """

    db_client = get_data_db_client(
        exchange=exchange,
        database_type=Database_Type(data_type.upper()),
        data_directory=data_directory,
    )
    # Call domain training data to split training and eval time range
    training_time_range, eval_time_range = splitting_training_and_eval_time_range(
        start_date=start_date,
        end_date=end_date,
        data_length=data_length,
        split_ratio=split_ratio,
        data_step=data_step,
    )

    def _save_data_to_storage(data_type: str, time_ranges: list[tuple]) -> int:
        with TrainingDataStorage(
            output_folder=os.path.join(output_folder, data_type),
            buffer_size=10000,
            datafile_prefix=f"data",
        ) as data_storage:
            for i, time_ranges in enumerate(time_ranges):
                # convert _start_date and _end_date to int in ms
                # logger.debug(f"Processing {i} data: {time_ranges}")
                _start_date, _end_date = time_ranges
                _start_date_ms = int(_start_date.timestamp() * 1000)
                _end_date_ms = int(_end_date.timestamp() * 1000)
                # Get  candles
                candles: pd.Dataframe = db_client.get_candles(
                    symbol=symbol,
                    from_time=_start_date_ms,
                    to_time=_end_date_ms,
                )
                # logger.debug("read Candles: %s", len(candles))

                # Resample training candles
                candles_sampled: pd.DataFrame = resample_timeframe(
                    data=candles,
                    tf=candle_size,
                )

                # Filter candles
                if len(candles_sampled) < min_candle_population:
                    continue

                candles_sampled["scenario"] = i

                # Save training candles
                # logger.debug("write Candles: %s", len(candles_sampled))
                data_storage.save_data(candles_sampled)
            data_storage.flush()
            logger.info(f"Written {data_type} data: {data_storage.written_rows}")
            return data_storage.written_rows
        pass

    num_training_rows_written = _save_data_to_storage(
        data_type=Training_Eval_Enum.TRAINING, time_ranges=training_time_range
    )
    num_eval_rows_written = _save_data_to_storage(
        data_type=Training_Eval_Enum.EVAL, time_ranges=eval_time_range
    )

    return num_training_rows_written, num_eval_rows_written
