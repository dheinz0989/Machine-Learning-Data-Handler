"""
This module defines a data class used to process features by time.
It now has only minor method which are intended to have more methods added by time. As of today (08.05.2020), it has the following methods

    - ``rolling_mean_plot``: a plot for visualizing time data
    - ``rel_plot``: a plot for visualizing time data

Furthermore, it has code to generate features out of date time data.
Note that all of those classes are using the BaseEstimator and TransformerMixin classes from the scikit-learn bases classes.
In order to use the ``fit_transform`` implemented by the scikit-learn, every single Feature Transformer needs to have a  ``fit`` and ``transform`` method.
The ``fit`` methods returns the object itself and is therefore not documented

"""
from __future__ import annotations
from datetime import datetime
from typing import Union, Sequence, Dict

import seaborn as sns
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# temporary hack to enable relative imports when script is not run within package
# see https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time?rq=1
# or https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py?rq=1

try:
    from baseattrs import BaseStats
except ImportError:
    from .baseattrs import BaseStats



__all__ = [
    "DateByAnalyzer",
    "UnixTimeStampConverter",
    "TimeFeatureBasic",
    "TimeDifference",
    "StringToTimeConverter",
]


class DateByAnalyzer(BaseStats):
    """
    A class which can be used to plot features (x_cols) by Time
    """

    def __init__(
        self,
        data: Union[str, DataFrame],
        x_cols: Union[str, Sequence[str]],
        date_cols: Union[str, Sequence[str]],
    ):
        """
        Initializes a Date By Analyzer object

        :param data: The source or Data Frame the source data comes from
        :param x_cols: the different columns which are to be analyzed by time
        :param date_cols: the date columns.
        """
        super().__init__(data)
        self.x_cols = x_cols if isinstance(x_cols, list) else [x_cols]
        self.date_cols = date_cols if isinstance(date_cols, list) else [date_cols]
        self.x = self.date_cols[0]
        self.date = self.x_cols[0]

    def rolling_mean_plot(self, n_roll: int = 6, **options: Dict):
        """
        Plots a rolling mean over time of the x column

        :param n_roll: number of Windows to take into the calculation of the mean
        :param options: plotting options
        :return:
        """
        # TODO include the optional plotting options
        data = self.data[[self.x, self.date]].rolling(n_roll).mean()
        sns.lineplot(data=data, palette="tab10", linewidth=2.5)
        # sns.lineplot(self.data[self.current_date], self.data[self.current_date], **options)

    def rel_plot(self, **options):
        """
        Generates a relationship plot between x column by the date.

        :param options:
        :return:
        """
        sns.relplot(x=self.x, y=self.date, data=self.data, **options)




class UnixTimeStampConverter(BaseEstimator, TransformerMixin):
    """
    Converts the Unix Time Stamp of a feature to a datetime Object
    """

    def __init__(self, features: Union[str, Sequence]):
        self.features = features if isinstance(features, list) else [features]
        self.feature_names = []

    def fit(self, X, y=None, **kwargs) -> UnixTimeStampConverter:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms a Unix Timestamp to a Pandas DataFrame ``datetime`` object

        :param X: The DataFrame whose columns is a UnixTimeStamp
        :return:
        """
        X = X.to_frame() if isinstance(X, pd.Series) else X
        for ts in self.features:
            X[ts] = pd.to_datetime(X[ts], unit="s")
        self.feature_names = X.columns.tolist()
        return X


class TimeDifference(BaseEstimator, TransformerMixin):
    """
    Returns the difference in time between two different time columns.
    """

    def __init__(
        self,
        x_time: str,
        y_time: str,
        days: bool = True,
        seconds: bool = False,
        microseconds: bool = False,
        nanoseconds: bool = False,
        components: bool = False,
    ):
        """
        Initializes a Time Difference Calculator object to fit_transform a DataFrame
        The following return columns can be retrieved:

            - ``days``: the difference of days between the two columns
            - ``seconds``: the difference of seconds between the two columns
            - ``microseconds``: the difference of microseconds between the two columns
            - `` nanoseconds``:  the difference of nanoseconds between the two columns
            - ``components``: the difference expressed in date time components

        :param x_time: first time column
        :type x_time: str
        :param y_time: second time column
        :type y_time: str
        :param days: a flag indicating if a column with the difference in ``days`` is returned
        :type days: bool
        :param seconds: a flag indicating if a column with the difference in ``seconds`` is returned
        :type seconds: bool
        :param microseconds: a flag indicating if a column with the difference in ``microseconds`` is returned
        :type microseconds: bool
        :param nanoseconds: a flag indicating if a column with the difference in ``nanoseconds`` is returned
        :type nanoseconds: bool
        :param components: a flag indicating if a column with the difference in ``datetime components`` is returned
        """
        self.x_time = x_time
        self.y_time = y_time
        self.days = days
        self.seconds = seconds
        self.microseconds = microseconds
        self.nanoseconds = nanoseconds
        self.components = components
        self.feature_names = []

    def fit(self, X, y=None, **kwargs) -> TimeDifference:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame by creating new columns yielding the time difference.
        Iterates over all boolean arguments and applies a transformation is the boolean flag is set to True.

        :param X: The DataFrame for which the time difference between x_time and y_time is yielded
        :return:
        """
        X = X.to_frame() if isinstance(X, pd.Series) else X
        if self.days:
            X["d_days" + self.x_time + "_" + self.y_time] = (
                X[self.x_time] - X[self.y_time]
            ).dt.days
        if self.seconds:
            X["d_seconds" + self.x_time + "_" + self.y_time] = (
                X[self.x_time] - X[self.y_time]
            ).dt.seconds
        if self.microseconds:
            X["d_ms" + self.x_time + "_" + self.y_time] = (
                X[self.x_time] - X[self.y_time]
            ).dt.microseconds
        if self.nanoseconds:
            X["d_ns" + self.x_time + "_" + self.y_time] = (
                X[self.x_time] - X[self.y_time]
            ).dt.nanoseconds
        if self.components:
            X["d_ns" + self.x_time + "_" + self.y_time] = (
                X[self.x_time] - X[self.y_time]
            ).dt.components
        self.feature_names = X.columns.tolist()
        return X


class StringToTimeConverter(BaseEstimator, TransformerMixin):
    """
    Converts a time column expressed as a string to a Datetime object
    """

    def __init__(self, features: Union[str, Sequence], format_string: str):
        """
        Initializes an object to convert string column to a datetime column

        :param features: the feature column whose column string is to be converted
        :type features:  Union[str, Sequence]
        :param format_string: the string format to be converted to a datetime object
        :type format_string: str
        """
        self.features = features if isinstance(features, list) else [features]
        self.format_string = format_string
        self.feature_names = []

    @staticmethod
    def as_date_time(date_str: str, format_string: str = "%Y-%m-%d") -> datetime:
        """
        Converts a date string to a Python datetime object. Format string is YYYY-MM-DD

        :param str date_str: a date in a string format
        :format_string: the string format
        :type format_string: str
        :return: the date in a datetime format
        :rtype: datetime

        """
        return datetime.strptime(date_str, format_string)

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame columns to an datetime object.

        :param X: The DataFrame for which the time difference between x_time and y_time is yielded
        :return:
        """
        # options here
        # assert isinstance(X, pd.DataFrame)
        X = X.to_frame() if isinstance(X, pd.Series) else X
        for feature in self.features:
            X[feature] = X[feature].apply(
                lambda c: self.as_date_time(c, self.format_string)
            )
        self.feature_names = X.columns.tolist()
        return X


class TimeFeatureBasic(BaseEstimator, TransformerMixin):
    """
    Derives Time features properties in the DataFra e column
    """

    # TODO Implement weekend
    def __init__(
        self,
        time_features: Union[str, Sequence],
        delete_original_column: bool = True,
        month: bool = True,
        year: bool = True,
        day: bool = True,
        dayofweek: bool = True,
        dayofyear: bool = True,
        hour: bool = True,
        minute: bool = True,
        week: bool = True,
        weekofyear: bool = True,
        quarter: bool = True,
        is_weekend: bool = True,
        is_leap_year: bool = True,
        is_month_end: bool = True,
        is_month_start: bool = True,
        is_quarter_start: bool = True,
        is_quarter_end: bool = True,
        is_year_start: bool = True,
        is_year_end: bool = True,
    ):
        """
        Initializes a Time Feature creator object.

        :param delete_original_column: a flag indicating if the original column is dropped from the resulting DataFrame
        :type delete_original_column: bool
        :param month: a flag indicating if the datetime's ``month`` is derived
        :type month: bool
        :param year: a flag indicating if the datetime's ``year`` is derived
        :type year: bool
        :param day: a flag indicating if the datetime's ``day`` is derived
        :type day: bool
        :param dayofweek: a flag indicating if the datetime's ``day of the week`` is derived
        :type dayofweek: bool
        :param dayofyear: a flag indicating if the datetime's ``day of the year`` is derived
        :type dayofyear: bool
        :param hour: a flag indicating if the datetime's ``hour`` is derived
        :type hour: bool
        :param minute: a flag indicating if the datetime's ``minute`` is derived
        :type minute: bool
        :param week: a flag indicating if the datetime's ``week`` is derived
        :type week: bool
        :param is_weekend: a flag indicating if the datetime is during a weekend
        :type is_weekend: bool
        :param weekofyear:  a flag indicating if the datetime's ``week of year` is derived
        :type weekofyear: bool
        :param quarter: a flag indicating if the datetime's ``quarter`` is derived
        :type quarter: bool
        :param is_leap_year: a flag indicating if the datetime's  derives, if it is a leap year
        :type is_leap_year: bool
        :param is_month_end: a flag indicating if the datetime's  derives, if it is the month's last day
        :type is_month_end: bool
        :param is_month_start: a flag indicating if the datetime's  derives, if it is the month's first day
        :type is_quarter_start: bool
        :param is_quarter_start: a flag indicating if the datetime's  derives, if it is the quarter's first day
        :type is_month_start: bool
        :param is_quarter_end: a flag indicating if the datetime's  derives, if it is the quarter's last day
        :type is_quarter_end: bool
        :param is_year_start: a flag indicating if the datetime's  derives, if it is the year's first day
        :type is_year_start: bool
        :param is_year_end: a flag indicating if the datetime's  derives, if it is the year's last day
        :type is_year_end: bool
        """

        self.time_features = (
            time_features if isinstance(time_features, list) else [time_features]
        )
        self.delete_original_column = delete_original_column
        self.month = month
        self.year = year
        self.day = day
        self.dayofweek = dayofweek
        self.dayofyear = dayofyear
        self.hour = hour
        self.minute = minute
        self.week = week
        self.weekofyear = weekofyear
        self.quarter = quarter
        self.is_weekend = is_weekend
        self.is_leap_year = is_leap_year
        self.is_month_end = is_month_end
        self.is_month_start = is_month_start
        self.is_quarter_start = is_quarter_start
        self.is_quarter_end = is_quarter_end
        self.is_year_start = is_year_start
        self.is_year_end = is_year_end
        self.feature_names = []

    def fit(self, X, y=None, **kwargs):
        return self

    @staticmethod
    def _is_weekend(feature):
        return feature.dt.dayofweek >= 5

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates over all boolean columns. If they are set to True, the respective feature is derived

        :param X: The DataFrame for which the different features a re derived
        :return:
        """
        X = X.to_frame() if isinstance(X, pd.Series) else X
        for ftr in self.time_features:
            if self.month:
                X[ftr + "_month"] = X[ftr].dt.month
            if self.year:
                X[ftr + "_year"] = X[ftr].dt.year
            if self.day:
                X[ftr + "_day"] = X[ftr].dt.day
            if self.dayofweek:
                X[ftr + "_dayofweek"] = X[ftr].dt.dayofweek
            if self.dayofyear:
                X[ftr + "_dayofyear"] = X[ftr].dt.dayofyear
            if self.week:
                X[ftr + "_week"] = X[ftr].dt.week
            if self.weekofyear:
                X[ftr + "_weekofyear"] = X[ftr].dt.weekofyear
            if self.hour:
                X[ftr + "_hours"] = X[ftr].dt.hour
            if self.minute:
                X[ftr + "_minute"] = X[ftr].dt.minute
            if self.quarter:
                X[ftr + "_quarter"] = X[ftr].dt.quarter
            if self.is_weekend:
                X[ftr + '_is_weekend'] = X[ftr].dt.dayofweek >= 5
            if self.is_leap_year:
                X[ftr + "_is_leap_year"] = X[ftr].dt.is_leap_year
            if self.is_month_end:
                X[ftr + "_is_is_month_end"] = X[ftr].dt.is_month_end
            if self.is_month_start:
                X[ftr + "_is_month_start"] = X[ftr].dt.is_month_start
            if self.is_quarter_start:
                X[ftr + "_is_quarter_start"] = X[ftr].dt.is_quarter_start
            if self.is_quarter_end:
                X[ftr + "_is_quarter_end"] = X[ftr].dt.is_quarter_end
            if self.is_year_start:
                X[ftr + "_is_year_start"] = X[ftr].dt.is_year_start
            if self.is_year_end:
                X[ftr + "_is_year_end"] = X[ftr].dt.is_year_end
            if self.delete_original_column:
                X.drop(ftr, axis=1, inplace=True)
        self.feature_names = X.columns.tolist()
        return X