from cryptomarketdata.port.db_client import get_data_db_client, Database_Type
from cryptomarketdata.utility import resample_timeframe
from datetime import datetime, timedelta
from ..domains.training_data import splitting_training_and_eval_time_range
import pandas as pd
from ..adapter.TrainingDataStorage import TrainingDataStorage
import os
from ..logging import get_logger

logger = get_logger(__name__)


def prepare_training_data_and_eval_from_parquet(
    exchange: str,
    symbol: str,
    data_directory: str,
    start_date: datetime,
    end_date: datetime,
    data_length: timedelta,
    split_ratio: float,
    output_folder: str,
    candle_size: str,
    min_candle_population: int,
    data_type: str = "PARQUET",
) -> tuple[str, str]:
    """Prepare training data and eval data from parquet file


    Args:

        exchange (str): exchange name
        symbol (str): symbol name
        data_directory (str): data directory
        start_date (datetime): start date
        end_date (datetime): end date
        data_length (timedelta): data length
        split_ratio (float): split ratio
        output_folder (str): output folder
        candle_size (str): candle size e.g. 15Min, 1H, D
        min_candle_population (int): min candle population
        data_type (Database_Type): data type
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
    )

    def _save_data_to_storage(data_type: str, time_ranges: list[tuple]) -> None:
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
            logger.info(f"Written {data_type} data: {data_storage.written_rows}")
        pass

    _save_data_to_storage(data_type="training", time_ranges=training_time_range)
    _save_data_to_storage(data_type="eval", time_ranges=eval_time_range)

    pass
