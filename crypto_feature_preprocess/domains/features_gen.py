from __future__ import annotations
from abc import ABCMeta, abstractmethod
import numpy as np
from .indicators import (
    calculate_log_price_change,
    calculate_simple_moving_average,
    calculate_rsi,
)
import pandas as pd

from cryptomarketdata.logging import get_logger

logger = get_logger(__name__)


class Feature(metaclass=ABCMeta):
    """feature class"""

    @abstractmethod
    def output_feature_array(self, normalize: bool = False) -> np.ndarray:
        """output array

        Returns:
            np.ndarray: array
        """
        pass

    def create_feature_from_raw_data_array(
        self, raw_data_array: np.ndarray, look_back: int
    ) -> np.ndarray:
        """create feature from raw array with lookback

        Args:
            raw_data_array (np.ndarray): raw array
            look_back (int): dimension of the feature, i.e. look back period

        Returns:
            np.ndarray: feature array
        """
        # Constract feature array of (N-look_back+1) x (look_back)
        feature_array = np.zeros((len(raw_data_array) - look_back + 1, look_back))

        for i in range(look_back):
            feature_array[:, look_back - 1 - i] = raw_data_array[
                i : i + len(raw_data_array) - look_back + 1
            ]
        return feature_array

    def feature_length(self, raw_data_length: int, look_back: int) -> tuple:
        return raw_data_length - look_back + 1


class Log_Price_Feature(Feature):
    """log price feature class"""

    def __init__(
        self, df_price: pd.Series, dimension: int, normalize_value: float = 0.02
    ):
        """initialize log price feature class

        Args:
            df_price (pd.Series): price series
            dimension (int): dimension of the feature, i.e. look back period
            normalize_value(float) : normalize the price change by this value
        """
        self.df_price = df_price
        self.dimension = dimension
        self.normalized_value: float = normalize_value

    def _calculate(self) -> pd.Series:
        """helper function calculate log price change from pandas series

        Returns:
            pd.Series: log price change
        """
        log_price_change = calculate_log_price_change(self.df_price)
        # drop nan
        log_price_change = log_price_change.dropna()
        return log_price_change

    def output_feature_array(self, normalize: bool = False) -> np.ndarray:
        """output array
            Note: for crossing feature, need to move the feature to left by one candle

        Args:
            normalize (bool, optional): Normalize the value. Defaults to False.

        Returns:
            np.ndarray: _description_
        """
        log_price_raw: pd.Series = self._calculate()

        # Constract feature array of (N-dimension) x (dimension)
        log_price_feature_array = self.create_feature_from_raw_data_array(
            raw_data_array=log_price_raw.values, look_back=self.dimension
        )
        # Normalize value
        if normalize:
            log_price_feature_array /= self.normalized_value

        return log_price_feature_array

    @property
    def shape(self) -> tuple:
        """shape of the feature array

        Returns:
            tuple: shape of the feature array
        """
        # minus one to account for price difference
        return (
            self.feature_length(len(self.df_price), self.dimension) - 1,
            self.dimension,
        )


class SMA_Cross_Feature(Feature):
    """calculate the cross of two SMA signals

    Args:
        Feature (_type_): _description_
    """

    def __init__(
        self, df_price: pd.Series, sma_window_1: int, sma_window_2: int, dimension: int
    ) -> None:
        """_summary_

        Args:
            df_price (pd.Series): price series
            sma_window_1 (int): sma windows length 1
            sma_window_2 (int): sma windows length 2
            dimension (int): dimension of the feature, i.e. look back period
        """
        self.df_price = df_price
        self.sma_window_1 = sma_window_1
        self.sma_window_2 = sma_window_2
        self.dimension = dimension

    def _calculate(self) -> np.ndarray:
        """calculate the cross over of two SMA signals

        Returns:
            np.ndarray: Cross over signals of two SMA
        """
        sma_1 = calculate_simple_moving_average(self.df_price, self.sma_window_1)
        sma_2 = calculate_simple_moving_average(self.df_price, self.sma_window_2)

        sma_cross = self._cross_over_lineA_above_lineB(sma_1, sma_2)

        # Remove invalid from sma_cross
        sma_cross = sma_cross[self.invalid_data_length :]
        return sma_cross

    def _cross_over_lineA_above_lineB(
        self, lineA: pd.Series, lineB: pd.Series
    ) -> pd.Series:
        """calculate the cross over of two lines:
            lineA cross above lineB
            it output True/False if there is lineA crossing above linB signal

        Args:
            lineA (pd.Series): pandas Series column with numeric type
            lineB (pd.Series): pandas Series column with numeric type

        Returns:
            pd.Series: Series of True or False
        """
        lineA_minus_lineB = lineA - lineB
        prev_lineA_minus_lineB = lineA_minus_lineB.shift(1)

        _cross_over = np.where(
            ((lineA_minus_lineB > 0) & (prev_lineA_minus_lineB < 0)), True, False
        )
        return _cross_over

    def output_feature_array(self, normalize: bool = False) -> np.ndarray:
        """output array
            each feature represents:
            (T, T-1, T-2, ..., T-dimensional+1)
        Returns:
            np.ndarray: array
        """
        # Constract feature array of (N-dimension) x (dimension)
        sma_cross: np.ndarray = self._calculate()
        sma_cross_feature_array = self.create_feature_from_raw_data_array(
            raw_data_array=sma_cross, look_back=self.dimension
        )

        return sma_cross_feature_array

    @property
    def invalid_data_length(self) -> int:
        """invalid data length

        Returns:
            int: invalid data length
        """
        return max(self.sma_window_1, self.sma_window_2) - 1

    @property
    def shape(self) -> tuple:
        """shape of the feature array

        Returns:
            tuple: shape of the feature array
        """

        return (
            self.feature_length(len(self.df_price), self.dimension)
            - self.invalid_data_length,
            self.dimension,
        )


class RSI_Feature(Feature):
    """calculate the RSI feature

    Args:
        Feature (_type_): _description_
    """

    def __init__(
        self,
        df_price: pd.Series,
        rsi_window: int,
        dimension: int,
        normalize_value: float = 25,
        offset: float = 50,
    ) -> None:
        """RSI feature
            rsi feature = (rsi - offset) / normalize_value
        Args:
            df_price (pd.Series): market price series
            rsi_window (int): rsi window
            dimension (int): dimension of the feature, i.e. look back period
            normalize_value (float, optional): normalize value. Defaults to 25.
            offset (float, optional): offset. Defaults to 50.
        """
        self.df_price = df_price
        self.rsi_window = rsi_window
        self.dimension = dimension
        self.normalized_value: float = normalize_value
        self.offset: float = offset

    def _calculate(self) -> pd.Series:
        """calculate the RSI feature

        Returns:
            pd.Series: RSI feature
        """
        rsi = calculate_rsi(df_price=self.df_price, window=self.rsi_window)
        # drop nan
        rsi = rsi.dropna()
        return rsi

    def output_feature_array(self, normalize: bool = False) -> np.ndarray:
        """output array

        Args:
            normalize (bool, optional): Normalize the value. Defaults to False.

        Returns:
            np.ndarray: array
        """
        # Constract feature array of (N-dimension) x (dimension)
        rsi_raw: pd.Series = self._calculate()
        rsi_feature_array = self.create_feature_from_raw_data_array(
            raw_data_array=rsi_raw.values, look_back=self.dimension
        )
        # Normalize value
        if normalize:
            rsi_feature_array = (
                rsi_feature_array - self.offset
            ) / self.normalized_value

        return rsi_feature_array

    @property
    def shape(self) -> tuple:
        """shape of the feature array

        Returns:
            tuple: shape of the feature array
        """
        return (
            self.feature_length(len(self.df_price), self.dimension)
            - self.rsi_window
            - 1,
            self.dimension,
        )
        # return len(self.df_price) - self.rsi_window - self.dimension, self.dimension
