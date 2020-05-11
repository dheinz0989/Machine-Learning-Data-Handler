"""
This module provides classes to generate a feature builder instance.
It imports Feature builder classes for special data as Geo Data or Time columns. It is intended to build meaningful features out of them
and return the resulting DataFrame
"""
from __future__ import annotations
from typing import Union, Sequence, Mapping, List
from collections import namedtuple
from copy import deepcopy

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd

# temporary hack to enable relative imports when script is not run within package
# see https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time?rq=1
# or https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py?rq=1
try:
    from baseattrs import BasicDataAttributes
    from utilities import Logger, Decorators
    from gis import GisBasicTransformer, GisDistance, GisCluster
    from date_and_time import (
        UnixTimeStampConverter,
        TimeDifference,
        TimeFeatureBasic,
        StringToTimeConverter,
    )
except ImportError:
    from .baseattrs import BasicDataAttributes
    from .utilities import Logger, Decorators
    from .gis import GisBasicTransformer, GisDistance, GisCluster
    from .date_and_time import (
        UnixTimeStampConverter,
        TimeDifference,
        TimeFeatureBasic,
        StringToTimeConverter,
    )

log = Logger.initialize_log()
accepted_features = ["time", "gis"]
__all__ = ["FeatureBuilder"]


class _PipelineFunctions:
    """
    A class which defines a set of pipeline functions. Other classes, which build pipelines by calling function inherit from this class and use its
    functionality
    """

    def __init__(self):
        """
        Only an empty list with pipeline steps is provided at the beginning
        """
        self.pipeline_steps = []

    def reverse_pipeline(self) -> _PipelineFunctions:
        """
        Reverses the items in the pipeline

        :return: the same object whose pipeline is reversed
        """
        log.info("Reversing pipeline")
        self.pipeline_steps = self.pipeline_steps[::-1]
        return self

    def display_pipeline(self) -> None:
        """
        Lists all steps in the pipeline

        """
        log.info(
            f"The current pipelines has the following steps {[step for step in self.pipeline_steps]}"
        )

    def flush_pipeline(self) -> _PipelineFunctions:
        """
        Clears the current pipeline and drops all entries from it

        :return: the same object whose pipeline is flushed
        """
        log.info(
            f"Resetting the current pipeline with {len(self.pipeline_steps)} steps"
        )
        self.pipeline_steps = []
        return self

    @staticmethod
    def _get_pipeline_tuple(value: str) -> namedtuple:
        """
        Minor helper function which returns a namedtuple with a step and an actions entry

        :param value: the named of the namedtuple
        :type value: str
        :return: a namedtuple with a step and actions entry
        :rtype: namedtuple
        """
        return namedtuple(value, ["step", "actions"])


class _TimeFeatureAdder(_PipelineFunctions):
    """
    Provides functionality to derive Time related features. It inherits from the _PipelineFunctions class and allows to add transformer steps
    to the pipelines. The steps can be wrapped in a parent class and executed to derive time features.
    """

    def __init__(self):
        """
        Only an empty list with pipeline steps is provided at the beginning
        """
        super().__init__()

    def add_unix_timestamp_converter(
        self, features: Union[str, Sequence]
    ) -> _TimeFeatureAdder:
        """
        Adds a UnixTimeStampConverter transformer to the current pipeline.

        :param features: a string or list of feature names who are Unix Time stamps and shall be converted to datetime
        :type features: Union[str, Sequence]
        :return: the same object by adding a unix time stamp converter to the feature pipeline
        """
        log.info(
            f"Adding a Unix timestamp convert for the following features {features}"
        )
        step = make_pipeline(UnixTimeStampConverter(features))
        wrapped_step = self._get_pipeline_tuple("unix_ts")
        self.pipeline_steps.append(wrapped_step("unix_ts", step))
        return self

    def add_time_diff(
        self,
        x_time: str,
        y_time: str,
        days: bool = True,
        seconds: bool = False,
        microseconds: bool = False,
        nanoseconds: bool = False,
        components: bool = False,
    ) -> _TimeFeatureAdder:
        """
        Adds a Time difference between two column calculator to the current pipeline.

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
        :return: the same object by adding a time difference calculator to the feature pipeline
        """
        log.info(f"Adding a time difference between {x_time} and {y_time}.")
        # TODO add in which units to the logger
        # TODO generally, change logging information to include all options
        step = make_pipeline(
            TimeDifference(
                x_time=x_time,
                y_time=y_time,
                days=days,
                seconds=seconds,
                microseconds=microseconds,
                nanoseconds=nanoseconds,
                components=components,
            )
        )
        wrapped_step = self._get_pipeline_tuple("time_diff")
        self.pipeline_steps.append(wrapped_step("time_diff", step))
        return self

    def add_basic_features(
        self,
        features: Union[str, Sequence],
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
    ) -> _TimeFeatureAdder:
        """
        Adds a basic time and date feature to the current pipeline

        :param features: a set of features for which basic time and date properties are derived
        :type features: Union[str, Sequence]
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
        :return: the same object whose pipeline has an additional basic date and time feature builder step
        """
        log.info(f'Adding basic time features for "{features[0]}"') if len(
            features
        ) == 1 else log.info(
            f"Adding basic time features for the following features: {features}"
        )
        if delete_original_column:
            log.warning(
                f"The column(s) {features} will be deleted after the transformation. Take care that it is not taken"
                f" in subsequent column transfer step or reverse the pipeline"
            )
        step = make_pipeline(
            TimeFeatureBasic(
                time_features=features,
                delete_original_column=delete_original_column,
                month=month,
                year=year,
                day=day,
                dayofweek=dayofweek,
                dayofyear=dayofyear,
                hour=hour,
                minute=minute,
                week=week,
                weekofyear=weekofyear,
                quarter=quarter,
                is_weekend=is_weekend,
                is_leap_year=is_leap_year,
                is_month_end=is_month_end,
                is_month_start=is_month_start,
                is_quarter_start=is_quarter_start,
                is_quarter_end=is_quarter_end,
                is_year_start=is_year_start,
                is_year_end=is_year_end,
            )
        )
        wrapped_step = self._get_pipeline_tuple("basic_datetime")
        self.pipeline_steps.append(wrapped_step("basic_datetime", step))
        return self

    def add_str_to_time(
        self, features: Union[str, Sequence], format_string: str
    ) -> _TimeFeatureAdder:
        """
        Adds a string to time converter to the current pipeline.

        :param features: the feature column whose column string is to be converted
        :type features:  Union[str, Sequence]
        :param format_string: the string format to be converted to a datetime object
        :type format_string: str
        """
        log.info(
            f"Adding a string converter for feature(s) {features} with format string '{format_string}'"
        )
        step = make_pipeline(
            StringToTimeConverter(features=features, format_string=format_string)
        )
        wrapped_step = self._get_pipeline_tuple("str_to_time")
        self.pipeline_steps.append(wrapped_step("str_to_time", step))
        return self


class _GisFeatureBuilder(_PipelineFunctions):
    """
     Provides functionality to derive Geo Information System (gis) related features. It inherits from the _PipelineFunctions class and allows to add transformer steps
    to the pipelines. The steps can be wrapped in a parent class and executed to derive time features.
    """

    def __init__(self, lat: str, lon: str):
        """
        Initializes a GIS Feature Builder object. It takes a lat and lon column and also inherits the pipeline steps list of its parent class

        :param lat: the column name of the latitude feature
        :type lat: str
        :param lon: the column name of the longitude feature
        :type: lon: str
        """
        super().__init__()
        self.lat = lat
        self.lon = lon
        log.info(
            f"The latitude and longitude columns are set to '{self.lat}' and '{self.lon}' upon object creation. "
            "You need to set them using ``set_lat`` and ``set_lon`` to ease execution "
        )

    def set_lat(self, lat: str) -> _GisFeatureBuilder:
        """
        Resets the latitude column

        :param lat: the latitude column name
        :type lat: str
        :return: the object with a rest of the latitude column
        """
        self.lat = lat
        return self

    def set_lon(self, lon: str) -> _GisFeatureBuilder:
        """
        Resets the longitude column

        :param lon: the longitude column name
        :type lon: str
        :return: the object with a rest of the longitude column
        """
        self.lon = lon
        return self

    def add_basic_gis(
        self, lat: str, lon: str, round_factor: int = 0, radians: bool = True
    ) -> _GisFeatureBuilder:
        """
        Adds a basic gis feature builder to the current pipeline.

        :param lat: the column name of the latitude column
        :type lat: str
        :param lon: the column name of the longitude column
        :type lon: str
        :param round_factor: if set and >0, it indicate the number of digits to round the geo data.
        :type round_factor: int
        :param radians: a flag indicating if the radians shall be derived
        :type radians: bool
        :return:
        """
        log.info(
            f"Adding a basic GIS feature builder for lat '{lat}' and lon '{lon}' including rounding to {round} and radians {radians} "
        )
        step = make_pipeline(
            GisBasicTransformer(
                lat=self.lat, lon=self.lon, round_factor=round_factor, radian=radians
            )
        )
        wrapped_step = self._get_pipeline_tuple("basic_gis")
        self.pipeline_steps.append(wrapped_step("basic_gis", step))
        return self

    def add_distance(
        self, lat_2: str = "", lon_2: str = "", label: str = "distance"
    ) -> _GisFeatureBuilder:
        """
        Adds a distance between two points calculator to the current pipeline

        :param lat_2: the column name of the second point's latitude column
        :type lat_2: str
        :param lon_2: the column name of the second point's longitude column
        :type lon_2: str
        :param label: the column label of the newly created column capturing the distance
        :type label: str
        :return:
        """
        log.info(
            f"Adding a basic GIS distance builder for coordinate 1 ('{self.lat}','{self.lon}') and  coordinate 2 ('{lat_2}','{lon_2}'). Saving the distance into column '{label}'"
        )
        step = make_pipeline(
            GisDistance(
                lat_1=self.lat, lon_1=self.lon, lat_2=lat_2, lon_2=lon_2, label=label
            )
        )
        wrapped_step = self._get_pipeline_tuple("distance")
        self.pipeline_steps.append(wrapped_step("distance", step))
        return self

    def add_clustering(
        self,
        kmeans: bool = True,
        kmeans_params: Mapping[str, ] = None,
        dbscan: bool = True,
        dbscan_params: Mapping[str, ] = None,
        birch: bool = True,
        birch_params: Mapping[str, ] = None,
        hdbscan: bool = True,
        hdbscan_params: Mapping[str, ] = None,
        agglomerative: bool = True,
        agglomerative_params: Mapping[str, ] = None,
    ) -> _GisFeatureBuilder:
        """

        :param kmeans: a flag indicating if the k-means clustering algorithm is applied
        :type kmeans: bool
        :param kmeans_params: indicates k-means clustering hyper parameters
        :type kmeans_params: dict
        :param dbscan:  a flag indicating if the DBSCAN clustering algorithm is applied
        :type dbscan: bool
        :param dbscan_params: indicates DBSCAN clustering hyper parameters
        :type dbscan_params: dict
        :param birch: a flag indicating if the BIRCH clustering algorithm is applied
        :type birch:  bool
        :param birch_params: indicates BIRCH clustering hyper parameters
        :type birch_params: dict
        :param hdbscan: a flag indicating if the HDBSCAN clustering algorithm is applied
        :type hdbscan:  bool
        :param hdbscan_params: indicates HDBSCAN clustering hyper parameters
        :type hdbscan_params: dict
        :param agglomerative: a flag indicating if the Agglomerative clustering algorithm is applied
        :type agglomerative:  bool
        :param agglomerative_params: indicates Agglomerative clustering hyper parameters
        :type agglomerative_params: dict
        :return:
        """
        log.info(f"Adding a set of Geo cluster algorithm ")
        step = make_pipeline(
            GisCluster(
                lat=self.lat,
                lon=self.lon,
                kmeans=kmeans,
                kmeans_params=kmeans_params,
                dbscan=dbscan,
                dbscan_params=dbscan_params,
                birch=birch,
                birch_params=birch_params,
                hdbscan=hdbscan,
                hdbscan_params=hdbscan_params,
                agglomerative=agglomerative,
                agglomerative_params=agglomerative_params,
            )
        )
        wrapped_step = self._get_pipeline_tuple("distance")
        self.pipeline_steps.append(wrapped_step("distance", step))
        return self


class FeatureBuilder(BasicDataAttributes):
    """
    Main class to generate a feature builder. It has methods to apply the classes defined in the module. It is intended
    to generate feature of special data type. At the moment, following feature classes are implemented:

        - time
        - gis

    In future version, other classes as text is going to be implemented

    Usage: the idea of this class is to have pipelines generator for the above-mentioned data types.
    In order to add and convert time features, the following implementation is used::

    > # Initialize a feature builder "builder"
    > # adding two generic time features
    > 1 builder.Time.add_...
    > 2 builder.Time.add_..
    > # and then wrap the pipeline and rut it via
    > 3 builder.wrap_and_run_current_pipeline()

    """

    # Implement a set features and operate on object attributes
    def __init__(
        self, data_source: Union[str, pd.DataFrame],
    ):
        """
        Initializes a feature builder object. As its parent classes, only a pointer to a file or a pandas DataFrame is required for initialization

        Other attributes are as follows

        The main feature builder classes:

        - ``Time``: is an attribute which allows to build pipeline for creating time related features
        - ``GIS``: is an attribute which allows to build pipeline for creating gis related features

        After applying feature transformation:

        - ``transformed_input``: is pandas DataFrame containing the derived features
        - ``transformed_features``: are all new columns derived within the feature creation

        Pipeline information:

        - ``wrapped_pipeline``: is an attribute, which contains all information when a pipeline of ``Time`` or ``GIS`` has been wrapped
        - ``_current_feature_family_in_pipeline``: gives information, to which feature class the current pipeline belongs to. Only time and gis is allowed at the moment

        :param data_source: the source of the data
        :type data_source: Union[str, pd.DataFrame]
        """
        super().__init__(data_source)
        self.Time = _TimeFeatureAdder()
        self.GIS = _GisFeatureBuilder("lat", "lon")
        self.transformed_input = None
        self.transformed_features = None
        self.wrapped_pipeline = None
        self._current_feature_family_in_pipeline = "time"
        self.__current_feature_mapping = {"time": self.Time, "gis": self.GIS}
        log.info(
            f"Initialized a FeatureBuilder Object. The current feature family in the pipeline is set to "
            f"{self._current_feature_family_in_pipeline}. Use ``set_current_pipeline_feature_family`` to reset it."
            f" Allowed arguments are {accepted_features} "
        )

    @Decorators.accepted_arguments_within_class_methods(accepted_features)
    def set_current_pipeline_feature_family(self, value: str) -> FeatureBuilder:
        """
        Resets the features of the current pipeline. Use this function to switch between different pipeline generators and wrap and run them

        :param value: a feature family
        :type value: str
        :return:
        """
        log.info(f'Setting current feature in_wrapped pipeline to "{value}"')
        self._current_feature_family_in_pipeline = value
        return self

    @staticmethod
    def get_transformer_feature_names(
        column_transformer: ColumnTransformer,
    ) -> List[str]:
        """
        Gets all generated feature names of a feature transformer column after column transformation

        :param column_transformer: a transformer of a column transformer pipeline
        :type column_transformer: ColumnTransformer
        :return: all feature names of the newly generated DataFrame:
        :rtype: list
        """
        output_features = []
        for name, pipe, features in column_transformer.transformers_:
            if name != "remainder":
                for i in pipe:
                    trans_features = []
                    if hasattr(i, "categories_"):
                        trans_features.extend(i.get_feature_names(features))
                    elif hasattr(i, "feature_names"):
                        trans_features.extend(i.feature_names)
                    else:
                        trans_features = features
                output_features.extend(trans_features)

        return output_features

    def wrap_pip(self) -> FeatureBuilder:
        """
        Wraps the pipeline of the feature whose value is currently set in the ``_current_feature_family_in_pipeline`` and wraps it into the
        wrapped pipeline attribute as a ColumnTransformer object

        :return:
        """
        feature_dim = self.__current_feature_mapping[
            self._current_feature_family_in_pipeline
        ]
        log.info(
            f'Wrapping all steps in the "{self._current_feature_family_in_pipeline}" pipeline to a transformer list.'
            f" The result is saved into the ``.wrapped_pipeline`` attribute"
        )
        if feature_dim.pipeline_steps:
            transformer_list = [
                (pip.step, pip.actions, self.all_feature_columns)
                for pip in feature_dim.pipeline_steps
            ]
            self.wrapped_pipeline = ColumnTransformer(transformer_list)
            feature_dim.flush_pipeline()
            return self

    def run_wrapped_pipeline(self) -> Union[FeatureBuilder, None]:
        """
        If a pipeline has been wrapped and saved into the wrapped_pipeline attribute, it takes this wrapped pipeline and executes it. It then returns
        a DataFrame after all transformation steps have been applied

        Caution: Temporary results are not saved within this pipeline. For instance, if two pipeline steps transform a given column and the
        second transformation steps depends on the transformation results of the first step, the steps might fall.

        Example::

        > # A first transformer converts a date in a string format to date format
        > 1 builder.Time.add_str_to_time('Month', "%Y-%m")
        > # A second transformer derives basic features of a datetime column.
        > # It therefore expects the column to be in a datetime format:
        > 2 builder.Time.add_basic_features('Month')
        > # Setting the wrapper to the Time feature, wrapping and running will result in an error
        > 3 builder.set_current_feature_dimension('time')
        > 4 builder.wrap_pip()
        > 5 builder.run_wrapped_pipeline()

        This happens because the result of the first transformer (changing the data type) is not taken into account by the second transformer

        In order to evade the error, the pipeline needs to be wrapped and run separately as follows::

        > 1 builder.Time.add_str_to_time('Month', "%Y-%m")
        > 2 builder.wrap_pip().run_wrapped_pipeline()
        > 3 builder.Time.add_basic_features('Month')
        > 4 builder.wrap_pip().run_wrapped_pipeline()

        :return:
        """
        if self.wrapped_pipeline:
            log.info(
                "Running a pipeline without saving temporary results. Applying the pipeline on the input data"
            )
            orig_data, src = (
                (self.transformed_input, "transformed_input")
                if self.transformed_input is not None
                else (self.input, "input")
            )
            log.info(f'The data source for the transformation is in attribute "{src}"')
            transformed_data = self.wrapped_pipeline.fit_transform(orig_data)
            self.transformed_features = self.get_transformer_feature_names(
                self.wrapped_pipeline
            )
            log.info(f"The derives feature columns are: {self.transformed_features}")
            log.info(f"Transformed_Data shape is {transformed_data.shape}")
            log.info(
                "Saving the transformed data into ``.transformed_input`` and the features into ``.transformed_features``"
            )
            self.transformed_input = pd.DataFrame(
                transformed_data,
                columns=self.get_transformer_feature_names(self.wrapped_pipeline),
            )
            return self
        else:
            log.warning(
                "The wrapped Pipeline is empty. You first need"
                " to add processing step to the pipeline and wrap the pipeline using the ``wrap_pip`` method"
            )
            return

    def wrap_and_run_current_pipeline(self):
        """
        Wraps the current pipeline steps and executes all steps.

        This function is saver to use as temporary results are saved and the following operations are performed on the temporary data.
        Therefore, two chaining transformations where one depends on the other will run through.

        This function is the recommend approach to run a pipeline in order to guarantee that all temporary results are saved in a tmp file taken into account.

        :return:
        """
        feature_dim = self.__current_feature_mapping[
            self._current_feature_family_in_pipeline
        ]
        steps = deepcopy(feature_dim.pipeline_steps)
        if steps:
            for pipe in steps:
                feature_dim.flush_pipeline()
                feature_dim.pipeline_steps.append(pipe)
                self.wrap_pip()
                self.run_wrapped_pipeline()
        else:
            log.warning(
                f"The current pipeline of feature dimension {self._current_feature_family_in_pipeline}"
            )
