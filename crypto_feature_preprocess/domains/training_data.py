# Prepare training data and testing data
#
# Path: src/crypto_feature_preprocess/domains/training_data.py
# Compare this snippet from tests/preprocess/test_feature.py:
import numpy as np
from datetime import datetime, timedelta
from random import shuffle

from ..logging import get_logger

logger = get_logger(__name__)

from zlib import crc32


def _is_eval_data_by_hash(date_time: datetime, split_ratio: float) -> bool:
    # Referencing from chapter 2 "Heads-on machine learning"
    return (
        crc32(np.int64(date_time.timestamp() * 1000000)) < (1 - split_ratio) * 2**32
    )


def splitting_training_and_eval_time_range(
    start_date: datetime,
    end_date: datetime,
    data_length: timedelta,
    split_ratio: float = 0.8,
    data_step: timedelta = timedelta(days=1),
) -> tuple[list[tuple], list[tuple]]:
    """splitting training and eval time range

    Args:
        start_date (datetime): start date
        end_date (datetime): end date
        data_length (timedelta): length of the data
        split_ratio (float, optional): split ratio. Defaults to 0.8.
        data_step (timedelta, optional): step of the data feature. Defaults to timedelta(days=1).

    Returns:
        tuple[list[tuple], list[tuple]]: training time range, eval time range
    """
    # Calculate the number of days
    num_of_data_vector = int((end_date - start_date - data_length) / data_step) + 1
    logger.info(f"num_of_data_vector: {num_of_data_vector}")
    # Calculate the number of training days
    num_training_vector = int(num_of_data_vector * split_ratio)
    logger.info(f"num_training_vector: {num_training_vector}")

    # Create data vector from start date to end date
    training_time_range = []
    eval_time_range = []
    for i in range(num_of_data_vector):
        start_date_vector = start_date + i * data_step
        end_date_vector = start_date_vector + data_length
        data_vector = (start_date_vector, end_date_vector)
        if _is_eval_data_by_hash(start_date_vector, split_ratio):
            eval_time_range.append(data_vector)
        else:
            training_time_range.append(data_vector)

    return training_time_range, eval_time_range
