from __future__ import annotations
from typing import NamedTuple
import pandas as pd
from enum import Enum
from dataclasses import dataclass
import numpy as np


class Feature_Enum(str, Enum):
    LOG_PRICE = "LOG_PRICE"
    SMA_CROSS = "SMA_CROSS"
    RSI = "RSI"


class Training_Eval_Enum(str, Enum):
    TRAINING = "training"
    EVAL = "eval"


@dataclass
class Feature_Definition:
    meta: dict
    data: NamedTuple


from typing import Union


@dataclass
class Feature_Output:
    metadata: list[Union[Feature_Definition, list]]
    time_index: np.ndarray
    feature_data: np.ndarray  # [N X (accm of feature dimension)] matrix of features

    pass


class Log_Price_Feature_Interface(NamedTuple):
    dimension: int
    normalize_value: float = 0.02


class SMA_Cross_Feature_Interface(NamedTuple):
    sma_window_1: int
    sma_window_2: int
    dimension: int


class RSI_Feature_Interface(NamedTuple):
    rsi_window: int
    dimension: int
    normalize_value: float = 25
    offset: float = 50
