"""
This module includes a class which defines basic attributes and methods for the Data Explorer, Preprocessor and FeatureBuilder
and all their custom classes. The respective modules import this class and inherit from it
"""
from __future__ import annotations
from typing import Union, Sequence, Mapping, Dict
import numpy as np
import pandas as pd
from pathlib import Path
from collections import namedtuple
import sys

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    Binarizer,
    MinMaxScaler,
    StandardScaler,
)
from sklearn.impute import SimpleImputer, KNNImputer
# temporary hack to enable relative imports when script is not run within package
# see https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time?rq=1
# or https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py?rq=1
try:
    from utilities import Logger
    from dataframeselector import (
        NameSelector,
        TypeSelector
    )
except ImportError:
    from .utilities import Logger
    from .dataframeselector import (
        NameSelector,
        TypeSelector
    )

log = Logger.initialize_log()

__all__ = ["BasicDataAttributes", "BaseStats"]


class BasicDataAttributes:
    """
    This class contains several attributes, methods and properties used within different classes.
    """

    # Defining different data types used within pandas DataFrames.
    dtype_num = np.number
    dtype_str = object
    dtype_bool = bool
    dtype_date = np.datetime64
    dtype_cat = "category"

    def __init__(
        self,
        data_source: Union[str, pd.DataFrame],
        target_name: str = "",
        features: Union[str, Sequence] = None,
    ):
        """
        Initialization function for Basic Data Attributes.

        :param data_source: a string with a file containing Data or a pandas DataFrame to be read.
        :type data_source: str or pd.DataFrame
        :param target_name: optionally specify the target_name column within the data set
        :param features: optionally pre define features which are to be kept upon initialization
        """
        self.data_source = data_source
        self.target_name = target_name
        self.features = features
        self.input = None
        self.target = None

    def ingest(self, **options: Mapping[str, ]) -> BasicDataAttributes:
        """
        Ingests the data source provided in the attribute and derives a pandas DataFrame out of it. The following data source are possible:

            - csv files
            - json files
            - parquet files
            - pickle files
            - pandas DataFrame

        When the data source is a string containing a pointer to a file, the respective file is read. If it is however a pandas DataFrame, the data is kept
        in a DataFrame.

        The pandas DataFrame is saved into the input attribute

        :param options:
        :return:
        """
        if isinstance(self.data_source, str):
            suffix = Path(self.data_source).suffix
            reader_mapping = {
                ".csv": pd.read_csv,
                ".json": pd.read_json,
                ".parquet": pd.read_parquet,
                ".pl": pd.read_pickle,
                ".pickle": pd.read_pickle,
            }
            log.info(
                f'Data format was found in the "{suffix}" format. The respective reader function is pandas\'s "{reader_mapping[suffix].__name__}"'
            )
            reader = reader_mapping[suffix]
            self.input = reader(self.data_source, **options)
            log.info(
                "The data was read successfully and can be accessed via the .input attribute."
            )

        elif isinstance(self.data_source, pd.DataFrame):
            log.info(
                f'Data format is already a pandas DataFrame. It is written into the .input attribute."'
            )
            self.input = pd.DataFrame(self.data_source)
            # log.info('Deleting the .data_source attribute to free memory. ')
            # del self.data_source
        else:
            raise NotImplementedError(
                "The data ingest function only accepts a pointer to a data file or a Pandas DataFrame as input"
            )
        if self.target_name:
            self.target = self.input[self.target_name].to_frame()
        if self.features:
            self.input = self.input[[self.target_name] + self.features]
        return self

    @property
    def numerical_features(self) -> Sequence[str]:
        """
        Returns all feature columns having a numerical data type, i.e. int and float

        :return: a list of all feature columns having a numerical data type, i.e. int and float
        :rtype: list
        """
        return list(self.input.select_dtypes(include=[self.dtype_num]).columns)

    @property
    def categorical_features(self) -> Sequence[str]:
        """
        Returns all feature columns having a categorical data type

        :return: a list of all feature columns having a categorical data type
        :rtype: list
        """
        return list(self.input.select_dtypes(include=[self.dtype_cat]).columns)

    @property
    def string_features(self) -> Sequence[str]:
        """
        Returns all feature columns having a object data type. In pandas, object data types are found to be string features. It is however possible,
        that these are other native Python objects as list, dict or tuples.

        :return: a list of all feature columns having an object data type
        :rtype: list
        """
        return list(self.input.select_dtypes(include=[self.dtype_str]).columns)

    @property
    def bool_features(self) -> Sequence[str]:
        """
        Returns all feature columns having a boolean data type

        :return: a list of all feature columns having a boolean data type
        :rtype: list
        """
        return list(self.input.select_dtypes(include=[self.dtype_bool]).columns)

    @property
    def date_features(self) -> Sequence[str]:
        """
        Returns all feature columns having a datetime data type

        :return: a list of all feature columns having a datetime data type
        :rtype: list
        """
        return list(self.input.select_dtypes(include=[self.dtype_date]).columns)

    @property
    def Features(self) -> Sequence[str]:
        """
        Can be used to reset the current DataFrame to a specified set of features

        :return:
        """
        return self.features

    @Features.setter
    def Features(self, values) -> None:
        """
        Setter method which can be used to reset the current DataFrame to a specified set of features

        :param values:
        :return:
        """
        values = values if isinstance(values, list) else [values]
        log.info(
            f"Resetting the .input DataFrame to contain the following column features: {values}"
        )
        self.features = values
        self.input = self.input[self.features]

    @property
    def all_feature_columns(self) -> Sequence[str]:
        """
         Returns all feature columns of the data frame

        :return: a list of all feature columns
        :rtype: list

        :return:
        """
        return list(self.input.columns)

    @property
    def feature_types(self) -> Sequence[str]:
        """
        Returns a list of type for all features.

        :return: list of type for all features
        :rtype: list
        """
        return self.input.dtypes

    def reset_feature_type(
        self, feature: str, dtype: Union[str, np.number, int, float]
    ):
        """
        This function can be used to reset the data type for a specified column

        :param feature: ''
        :param dtype:
        :return:
        """
        self.input = (
            self.input.to_frame() if isinstance(self.input, pd.Series) else self.input
        )
        assert dtype in [str, np.number, int, float, "category"]
        assert feature in list(self.input.columns)
        try:
            self.input[feature] = self.input[feature].astype(dtype)
        except Exception as e:
            print(e)
        # TODO: implement logic for date time.

    @staticmethod
    def convert_nested_dictionary_class(d: Dict) -> Dict:
        """
        Takes a dictionary and converts a string values to the corresponding Python classes

        :param d: a dictionary holding values
        :return: the dictionary whose key values are now pointing to the value's corresponding classes
        """

        def str_to_class(classname):
            return getattr(sys.modules[__name__], classname)

        for k, v in d.items():
            if isinstance(v, dict):
                BasicDataAttributes.convert_nested_dictionary_class(v)
            else:
                d[k] = str_to_class(v)
        return d


class BaseStats:
    def __init__(self, data: Union[str, pd.DataFrame]):
        """
        Initialization of the basic class. It accepts a pandas DataFrame as input.

        :param data: the base DataFrame
        :rtype data: pd.DataFrame
        """
        self.data = data
        self.n_observation = len(data)

    def get_missing_values(self, column: str) -> namedtuple:
        """
        Returns the amount of missing values for a given column. Returns a namedtuple including the absolute and relative value of missing values.

        :param column: a string of the DataFame column
        :return: a namedtuple with absolute and relative amount of missing values
        :rtype: namedtuple
        """

        absolute = self.data[column].isnull().sum()
        perc = round(len(absolute) / self.n_observation, 2)
        missing = self._abs_rel_namedtuple_creator("missing")
        log.info(f'Calculating missing values for "{column}": about {perc * 100} %')
        return missing(absolute, perc)

    def get_stats(self) -> pd.DataFrame:
        """
        Returns the summary statistics of the pd.DataFrame

        :return: an overview of the descriptive statistics
        """
        summary = self.data.describe().T
        log.info(f"Summary statistics: \n{summary}")
        return summary

    def get_unique_values(self, column: str) -> namedtuple:
        """
        Returns the amount of unique values for a given column. Returns a namedtuple including the absolute and relative value of missing values.

        :param column: a string of the DataFame column
        :return: a namedtuple with absolute and relative amount of unique values
        :rtype: namedtuple
        """
        absolute = self.data[column].unique()
        perc = round(len(absolute) / self.n_observation, 2)
        unique = self._abs_rel_namedtuple_creator("unique")
        log.info(f'Calculating unique values for "{column}": about {perc * 100} %')
        return unique(absolute, perc)

    @staticmethod
    def _abs_rel_namedtuple_creator(value: str) -> namedtuple:
        """
        Returns a namedtuple with an absolute and percent attribute

        :param value: the namedtuple name
        :return: a namedtuple with absolute and percent values
        :rtype: namedtuple
        """
        return namedtuple(value, ["absolute", "percent"])

    def get_size_in_memory(self, column: str) -> str:
        """
        Returns the amount of memory consumed by the given column

        :param column: a string of the DataFame column
        :return: the amount of memory consumed by the columns
        :rtype: str
        """
        mem_size = self.data[column].memory_usage()
        for x in ["bytes", "KB", "MB", "GB", "TB", "PB", "EX"]:
            if mem_size < 1024.0:
                size = "%3.2f %s" % (mem_size, x)
                log.info(f'Memory usage of column "{column}": {size}')
                return size
            mem_size /= 1024.0
