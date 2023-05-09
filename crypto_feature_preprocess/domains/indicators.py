import pandas as pd
import numpy as np


def calculate_log_price_change(df_price: pd.Series) -> pd.Series:
    """calculate log price change from pandas series

    Args:
        df_price (pd.Series): price series

    Returns:
        pd.Series: log price change
    """
    log_price_change = np.log(df_price / df_price.shift(1))

    return log_price_change


def calculate_simple_moving_average(df_price: pd.Series, window: int) -> pd.Series:
    """calculate simple moving average from pandas series

    Args:
        df_price (pd.Series): price series
        window (int): window size

    Returns:
        pd.Series: simple moving average
    """
    sma = df_price.rolling(window).mean()

    return sma


def calculate_rsi(df_price: pd.Series, window: int) -> pd.Series:
    """calculate relative strength index from pandas series

    Args:
        df_price (pd.Series): price series
        window (int): window size

    Returns:
        pd.Series: relative strength index
    """
    delta = df_price.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=window - 1, adjust=True, min_periods=window).mean()
    ma_down = down.ewm(com=window - 1, adjust=True, min_periods=window).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))

    return rsi
