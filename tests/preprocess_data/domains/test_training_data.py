from crypto_feature_preprocess.domains.training_data import (
    splitting_training_and_eval_time_range,
)
from datetime import datetime, timedelta
from crypto_feature_preprocess.logging import get_logger

logger = get_logger(__name__)


def test_splitting_training_and_eval_time_range() -> None:
    num_of_data_vector = 100
    data_length_days: int = 7
    data_step: int = 1
    start_date = datetime(2021, 1, 1)
    data_length = timedelta(days=data_length_days)
    end_date = start_date + timedelta(
        days=num_of_data_vector * data_step + data_length_days - 1
    )

    split_ratio = 0.8

    (training_time_range, eval_time_range) = splitting_training_and_eval_time_range(
        start_date=start_date,
        end_date=end_date,
        data_length=data_length,
        split_ratio=split_ratio,
        data_step=timedelta(days=data_step),
    )

    num_training_vector = len(training_time_range)
    assert abs((num_training_vector / num_of_data_vector) - split_ratio) < 0.1
    assert len(eval_time_range) == num_of_data_vector - num_training_vector

    pass
