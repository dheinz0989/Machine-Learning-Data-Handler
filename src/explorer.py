"""
This module provides a class to generate a data explorer object. It can handle different kinds of inputs and
generate quick overview, statistics and plots. It includes some subclasses to generate views and statistics for
different types of features. These include:

    - Numerical
    - Categorical
    - Mixed Types
    - GIS (Geo Data)
    - Numerical by Time
"""
from __future__ import annotations
from typing import Sequence, Union, Mapping
from collections import namedtuple
from math import ceil, sqrt

import seaborn as sns
import pandas as pd

# temporary hack to enable relative imports when script is not run within package
# see https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time?rq=1
# or https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py?rq=1
try:
    from baseattrs import BasicDataAttributes, BaseStats
    from gis import GisAnalyzer, GisAnalyzerWithClusterLabel
    from date_and_time import DateByAnalyzer
    from utilities import Logger
    from dataframeselector import TypeSelector, NameSelector
except ImportError:
    from .baseattrs import BasicDataAttributes, BaseStats
    from .gis import GisAnalyzer, GisAnalyzerWithClusterLabel
    from .date_and_time import DateByAnalyzer
    from .utilities import Logger
    from .dataframeselector import TypeSelector, NameSelector

log = Logger.initialize_log()


class NumericalAnalyzer(BaseStats):
    """
    Provides basic visualization for numerical features
    """

    def __init__(self, data: Union[str, pd.DataFrame]):
        """
        Initializes a NumericalAnalyzer object. Takes data file name or a pandas DataFrame as input

        :param data: link to a file or a pandas DataFrame:
        :type data: Union[str, pd.DataFrame]
        """
        super().__init__(data)

    def pair_plot(self, **options: Mapping[str, ]):
        """
        Generates a pair plot of all data

        :param options: options settings the plotter function
        :type options: Mapping
        :return:
        """
        # TODO include options in function
        sns.pairplot(self.data, **options)

    def corr(self, **options):
        """
        Generates a correlation heat map of all correlation between the data's features

        :param options: options settings the plotter function
        :type options: Mapping
        """
        # TODO include options in function
        sns.heatmap(self.data.corr())


class CategoricalAnalyzer(BaseStats):
    """
    Provides basic statistics and visualization for categorical features
    """

    def __init__(self, data: Union[str, pd.DataFrame]):
        """
        Initializes a CategoricalAnalyzer object. Takes data file name or a pandas DataFrame as input

        :param data: link to a file or a pandas DataFrame
        :type data: Union[str, pd.DataFrame]
        """
        super().__init__(data)

    def frequencies(self) -> Mapping[str, pd.DataFrame]:
        """
        For each column, calculates the different frequencies and saved them in a file

        :return: a dictionary with each column name as key and the frequencies as value
        """
        log.info("Examining frequency of values...")
        out = {}
        for column in list(self.data.columns):
            out[column] = self.data[column].value_counts()
            log.info(f"{column}:{out[column]}")
        return out

    def cat_bar_plot(self, **options) -> None:
        """
        Generates a bar plot for all frequency values

        :param options: options settings the plotter function
        :type options: Mapping
        """
        sns.barplot(self.data, **options)


class MixedDataAnalyser(NumericalAnalyzer, CategoricalAnalyzer):
    """
    Initializes an object to handle numerical and categorical features. At the current state, the code only consists of a prototype

    """

    def __init__(self, data, nums, group):
        super().__init__(data)
        self.nums = nums
        self.group = group


class DataExplorer(BasicDataAttributes):
    """
    Initializes a Data Explorer object. This object can be used to quickly generate insights of different features.

    """

    def __init__(
        self, data_source: Union[str, pd.DataFrame], target: str = "",
    ):
        """
        Initializes a Data Explorer Object. Takes a pandas DataFrame or a pointer to a data file and allows to quickly analyze it

        :param data_source: link to a file or a pandas DataFrame
        :type data_source: Union[str, pd.DataFrame]
        :param target: target column which is the interest of ML modeling
        :type target: str
        """
        super().__init__(data_source)
        self.target_name = target
        self._gis_features = []
        self._gis_data = None
        self._gis_data_with_labels = None
        self._data_grouped = None
        self._date_by_data = None
        self.__dtypes = [
            self.dtype_date,
            self.dtype_str,
            self.dtype_bool,
            self.dtype_cat,
            self.dtype_num,
        ]
    # TODO change implementation from properties to classes

    @property
    def Numerical_Data(self) -> NumericalAnalyzer:
        """
        Selects all numerical values and uses those as input values to create a Numerical_Data processor object.

        :return: a numerical data analyzer
        """
        if self.len_numerical_features:
            val = NameSelector(self.numerical_features).fit_transform(self.input)
            log.info(
                f"Setting numerical DataFrame of size {self.len_numerical_features} with the following columns: {self.numerical_features}"
            )
            return NumericalAnalyzer(val)

    @property
    def Data_Mixed_Types(self) -> DataExplorer:
        """
        A property which returns a data set of different types. This can be used to analyze relationships between different features of different types.

        :return:
        """
        return self._data_grouped

    @Data_Mixed_Types.setter
    def Data_Mixed_Types(self, dtype: Union[Sequence, str]):
        """
        Setter property for the data mixed types. It initializes a Data Mixed Tye Object. There are several ways to set it:

            1) as a tuple: in this case, for both arguments are made a check if the provided argument is a data type. If yes, all columns of this data type
            are provided. If not, all columns found in the respective arguments are kept. Example to mix receive a Mixed Data Type for all boolean columns
            on the one side and the columns "income" and "revenue" on the other side:

            > # Assume an ``explorer`` object named "explorer"
            >  explorer.Data_Mixed_Types = explorer.dtype_bool, ["income", "revenue"]

            Or to get all boolean and numeric types:

            > # Assume an ``explorer`` object named "explorer"
            >  explorer.Data_Mixed_Types = explorer.dtype_bool, explorer.dtype_num

            2) as a string: in this case, it is assumed, that the first column type is numerical, i.e. it is the default value. The second column can be a string that
            indicates either all columns of a given data type if the parameter is a valid data type or a single column. For example, to gain all numerical
            features by the boolean column "Male":

            > # Assume an ``explorer`` object named "explorer"
            >  explorer.Data_Mixed_Types = "Male"

            Or

            > # Assume an ``explorer`` object named "explorer"
            >  explorer.Data_Mixed_Types = bool

            3) as a list: in this case, it is assumed, that the first column type is numerical, i.e. it is the default value. The list indicates a list of features
            to be kept as second feature dimension. Example to mix receive a Mixed Data Type for all numerical columns on the one side and the columns "gender" and "married"
            on the other side:

            > # Assume an ``explorer`` object named "explorer"
            >  explorer.Data_Mixed_Types = ["gender", "married"]

        :param dtype: As described in the description, different ways exist to initialize call this object
        :type dtype: Union[Sequence, str]
        :return: a Data Mixed Type object
        """
        # first case, the provided argument is a tuple. the two entries are taken as input for the respective data type/ data column names
        if isinstance(dtype, tuple):
            log.info(
                f'Received the following two inputs "{dtype[0]}" and "{dtype[1]}". Deriving the data frame to be generated out of it. '
            )
            subset_1, subset_2 = dtype[0], dtype[1]
            data_1 = (
                TypeSelector(subset_1)
                if subset_1 in self.__dtypes
                else NameSelector(subset_1)
            )
            df_1 = data_1.fit_transform(self.input)
            data_2 = (
                TypeSelector(subset_2)
                if subset_2 in self.__dtypes
                else NameSelector(subset_2)
            )
            df_2 = data_2.fit_transform(self.input)
            result = pd.concat([df_1, df_2], axis=1)
            self._data_grouped = MixedDataAnalyser(
                result, list(df_1.columns), list(df_2.columns)
            )

        # second case: the provided argument is a string. The derived object is the column name indicated by the string. The second dimension is numerical features
        elif isinstance(dtype, str):
            log.info(
                f"Received a single input. The first data type dimension is automatically set to numerical. "
                f'The second is derived from the input "{dtype}"'
            )
            num = TypeSelector(self.dtype_num)
            group = (
                TypeSelector(dtype) if dtype in self.__dtypes else NameSelector(dtype)
            )
            data_num = num.fit_transform(self.input)
            data_grouped = group.fit_transform(self.input)
            result = pd.concat([data_num, data_grouped], axis=1)
            self._data_grouped = MixedDataAnalyser(
                result, self.numerical_features, list(data_grouped.columns)
            )

        # third case: the provided argument is a list: the entries in the list are the feature which are kept. The second dimension is numerical features.
        elif isinstance(dtype, list):
            log.info(
                f"Received a single input containing a list of features. The first data type dimension is automatically set to numerical. "
                f'The second is derived from the input "{dtype}"'
            )
            num = TypeSelector(self.dtype_num)
            group = NameSelector(dtype)
            data_num = num.fit_transform(self.input)
            data_grouped = group.fit_transform(self.input)
            result = pd.concat([data_num, data_grouped], axis=1)
            self._data_grouped = MixedDataAnalyser(
                result, self.numerical_features, list(data_grouped.columns)
            )
            # grouper = NameSelector(dtype)

        log.info(
            f"Setting Grouped DataFrame of size {len(self._data_grouped.data)} where the first feature dimension contains columns"
            f": {list(self._data_grouped.nums)} and the second dimension {list(self._data_grouped.group)}"
        )

    @property
    def Categorical_Data(self) -> CategoricalAnalyzer:
        """
        A property which initializes a Categorical Analyzer for categorical values.
        Takes all categorical features and uses them in a Categorical Analyzer object.

        :return: A Categorical Data Analyzer
        :rtype: CategoricalAnalyzer
        """
        if self.len_categorical_features:
            val = NameSelector(self.categorical_features).fit_transform(self.input)
            log.info(
                f"Created categorical DataFrame of size {self.len_categorical_features} with the following columns: {self.categorical_features}"
            )
            return CategoricalAnalyzer(val)

    @property
    def Boolean_Data(self) -> CategoricalAnalyzer:
        """
        A property which initializes a Categorical Analyzer for boolean values.
        Takes all boolean columns and uses them in a Categorical Analyzer object.

        :return: A Categorical Analyzer Object containing boolean values
        :rtype: CategoricalAnalyzer
        """
        if self.len_bool_features:
            val = NameSelector(self.bool_features).fit_transform(self.input)
            log.info(
                f"Created boolean DataFrame of size {self.len_bool_features} with the following columns: {self.bool_features}"
            )
            return CategoricalAnalyzer(val)

    @property
    def String_Data(self):
        """
        A property which initializes a String Data Analyzer object. Takes all string columns and uses them in a Categorical Analyzer object.

        :return: a Categorical Analyzer with all string features
        :rtype: CategoricalAnalyzer
        """
        if self.len_string_features:
            val = NameSelector(self.string_features).fit_transform(self.input)
            log.info(
                f"Created string DataFrame of size {self.len_string_features} with the following columns: {self.string_features}"
            )
            return CategoricalAnalyzer(val)

    @property
    def Date_By_Data(self):
        """
        A property which returns a Data By Date object. This can be used to analyze columns by time.

        :return:
        """
        return self._date_by_data

    @Date_By_Data.setter
    def Date_By_Data(self, dtype: Union[float, int, str, Union]) -> None:
        """
        A setter for the Data by date. It takes a column type and checks if the date_features are not empty.
        It then creates a Data by Date object for the provided data type.

        The provided argument can either be a data type defined in the class or a str/list of columns which are found in the DataFrame

        :param dtype: the data for which data shall be analyzed along the date column
        :type dtype: one of the classes defined data types
        :return: a date_by_Analyzer object
        """
        if self.len_date_features:
            date_data = NameSelector(self.date_features).fit_transform(self.input)
            x_transformer = (
                TypeSelector(dtype) if dtype in self.__dtypes else NameSelector(dtype)
            )
            x_data = x_transformer.fit_transform(self.input)
            result = pd.concat([date_data, x_data], axis=1)
            log.info(
                f"Created Date DataFrame of size {self.len_date_features} with the following columns: {self.date_features}"
            )
            self._date_by_data = DateByAnalyzer(
                result, list(x_data.columns), self.date_features
            )

    @property
    def GIS_Data(self) -> GisAnalyzer:
        """
        A property which initializes a GIS Analyzer by taking the first and second entry of the gis feature list.

        :return:
        """
        if self.len_gis_features >= 2:
            data = NameSelector(self._gis_features).fit_transform(self.input)
            log.info(
                f"Setting the GIS DataFrame of size {len(data)} with {self._gis_features[0]} as the latitude column and {self._gis_features[1]} as the longitude column."
                f"Note that you can reset them if necessary"
            )
            return GisAnalyzer(data, self._gis_features[0], self._gis_features[1])
        else:
            log.error(
                "No features have been defined as GIS data. You first need to set them using gis_features"
            )

    @property
    def Gis_DataWithLabel(self):
        """
        A property which returns all gis features with a corresponding label. This is intended to check for clustering algorithms results

        :return: a GisDatawithLabel object to analyze geo data and corresponding labels
        """
        return self._gis_data_with_labels

    @Gis_DataWithLabel.setter
    def Gis_DataWithLabel(self, labels: Union[str, Sequence]) -> None:
        """
        The setter for the GIS with label feature property.

        The label argument is either a data type, for which all features are selected or a str/list of labels of features, which are kept

        :param labels: indicates the labels for the GisDataWithLabel object
        :type labels:  Union[str, Sequence]
        :return:
        """
        data = NameSelector(self._gis_features).fit_transform(self.input)
        label_transformer = (
            TypeSelector(labels) if labels in self.__dtypes else NameSelector(labels)
        )
        label_data = label_transformer.fit_transform(self.input)
        result = pd.concat([data, label_data], axis=1)
        self._data_grouped = GisAnalyzerWithClusterLabel(
            result, self._gis_features[0], self._gis_features[1], labels
        )

    @property
    def GIS_features(self) -> Sequence[str]:
        """
        A property which returns all gis features of the DataFrame

        :return: a list of GIS features
        """
        return self._gis_features

    @GIS_features.setter
    def GIS_features(self, values: Union[Sequence, str]) -> None:
        """
        The setter for the GIS feature property.

        :param values: either a single string indicating the geo data column or a list of columns with geo data. Recommended way is to provide a list with
        lat and lon columns
        :type values: Union[Sequence, str]
        :return:
        """
        self._gis_features = values

    @property
    def len_gis_features(self):
        """
        Returns the length of all GIS features.

        :return: the length of all GIS features
        :rtype: int
        """
        return len(self._gis_features)

    @property
    def len_string_features(self) -> int:
        """
        Returns the length of all string features

        :return: the length of all string features
        :rtype: int
        """
        return len(self.string_features)

    @property
    def len_categorical_features(self):
        """
        Returns the length of all categorical features

        :return: the length of all categorical features
        :rtype: int
        """
        return len(self.categorical_features)

    @property
    def len_numerical_features(self):
        """
        Returns the length of all numerical features

        :return: the length of all numerical features
        :rtype: int
        """
        return len(self.numerical_features)

    @property
    def len_bool_features(self):
        """
        Returns the length of all boolean features

        :return: the length of all boolean features
        :rtype: int
        """
        return len(self.bool_features)

    @property
    def len_date_features(self):
        """
        Returns the length of all date features

        :return: the length of all date features
        :rtype: int
        """
        return len(self.date_features)

    @staticmethod
    def plot_grid(number: int) -> namedtuple:
        """
        A function used to determine the number of grids when doing multiple graphics.

        :param number: number of graphics to be performed
        :type number: int
        :return: a namedtuple indicating the number of rows and number of columns
        """
        rows = ceil(sqrt(number))
        columns = ceil(number / rows)
        plots = namedtuple("plot", ["n_rows", "n_columns"])
        return plots(rows, columns)
