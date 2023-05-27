from __future__ import annotations
from typing import NamedTuple
import pandas as pd
from enum import Enum
from dataclasses import dataclass


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
