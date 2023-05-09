import pytest

import pandas as pd
from datetime import datetime, timedelta
import numpy as np


@pytest.fixture
def get_test_ascending_mkt_data() -> pd.DataFrame:
    def _get_data(dim: int = 10, step: int = 100):
        # Generate pandas dataframe of timestamps records
        df = pd.DataFrame(
            data={
                "timestamp": pd.date_range(
                    start=datetime.now(), periods=dim, freq=timedelta(hours=1)
                ),
                "inx": np.arange(dim),
            }
        )
        df.set_index("timestamp", inplace=True, drop=True)
        df["close"] = df["inx"] * step + 1000

        df["price_movement"] = df["close"].diff()
        return df
        pass

    return _get_data


@pytest.fixture
def get_test_descending_mkt_data() -> pd.DataFrame:
    def _get_data(dim: int = 10, step: int = 100):
        # Generate pandas dataframe of timestamps records
        df = pd.DataFrame(
            data={
                "timestamp": pd.date_range(
                    start=datetime.now(), periods=dim, freq=timedelta(hours=1)
                ),
                "inx": np.arange(dim),
            }
        )
        df.set_index("timestamp", inplace=True, drop=True)
        df["close"] = (dim - df["inx"]) * step + 1000

        df["price_movement"] = df["close"].diff()
        return df
        pass

    return _get_data


@pytest.fixture
def get_test_decending_then_ascending_mkt_data() -> pd.DataFrame:
    def _get_data(dim: int = 100, step: int = 100):
        # Generate pandas dataframe of timestamps records
        df = pd.DataFrame(
            data={
                "timestamp": pd.date_range(
                    start=datetime.now(), periods=dim, freq=timedelta(hours=1)
                ),
                "inx": np.arange(dim),
            }
        )
        # mid point of the data
        mid = int(dim / 2)
        # Create nunmpy array of dim in length
        arr = np.arange(dim)
        for i in range(mid):
            arr[i] = (mid - i) * step + 1000
        for i in range(mid, dim):
            arr[i] = (i - mid) * step + 1000
        df.set_index("timestamp", inplace=True, drop=True)
        # Assign numpy array arr to df["close"]
        df["close"] = arr
        df["price_movement"] = df["close"].diff()
        return df

    return _get_data
