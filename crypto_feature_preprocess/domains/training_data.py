# Prepare training data and testing data
#
# Path: src/crypto_feature_preprocess/domains/training_data.py
# Compare this snippet from tests/preprocess/test_feature.py:
import numpy as np
from datetime import datetime, timedelta
from random import shuffle

from ..logging import get_logger

logger = get_logger(__name__)


def splitting_training_and_eval_time_range(
    start_date: datetime,
    end_date: datetime,
    data_length: timedelta,
    split_ratio: float = 0.8,
) -> tuple[list[tuple], list[tuple]]:
    """splitting training and eval time range

    Args:
        start_date (datetime): start date
        end_date (datetime): end date
        data_length (timedelta): length of the data
        split_ratio (float, optional): split ratio. Defaults to 0.8.

    Returns:
        tuple[list[tuple], list[tuple]]: training time range, eval time range
    """
    # Calculate the number of days
    num_of_data_vector = int((end_date - start_date) / data_length)
    # Calculate the number of training days
    num_training_vector = int(num_of_data_vector * split_ratio)
    # Calculate the number of eval days
    num_eval_vector = int(num_of_data_vector - num_training_vector)

    # Create data vector from start date to end date
    data_collections = []
    for i in range(num_of_data_vector):
        start_date_vector = start_date + i * data_length
        end_date_vector = start_date + (i + 1) * data_length
        data_vector = (start_date_vector, end_date_vector)
        data_collections.append(data_vector)

    # Splitting training and eval time range
    # randomly pick num_training_vector from data_collections
    shuffle(data_collections)
    training_time_range = data_collections[:num_training_vector]

    # eval time range is the rest of data_collections
    eval_time_range = data_collections[num_training_vector:]
    return training_time_range, eval_time_range
