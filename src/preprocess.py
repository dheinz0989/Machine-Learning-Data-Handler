"""
This module provides a data class which can be used to perform standard pre_processing steps
It inherits data attributes from the BasicDataAttributes class and extends them by adding
Preprocessing methods, attributes and properties.
It requires a config file in yaml format which adds a mapping for dictionary keywords and corresponding pre processor classes

"""
#import sys
#sys.path.append("..")
import numpy as np
import pandas as pd
from collections import namedtuple

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# temporary hack to enable relative imports when script is not run within package
# see https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time?rq=1
# or https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py?rq=1
try:
    from utilities import Logger
    from baseattrs import BasicDataAttributes
except ImportError:
    from .utilities import Logger
    from .baseattrs import BasicDataAttributes

log = Logger.initialize_log()


class DataPreProcessor(BasicDataAttributes):
    """
    A class used to perform pre-processing steps to the input data.
    """

    def __init__(
        self, data_source, target_name, config, features=None,
    ):
        """
        Initializes a Data Preprocessor object. It requires a data source which can either be a string to a file or a pandas DataFrame.
        Furthermore, it requires a target name for the target to predict.

        The last parameters is a dictionary containing a mapping to preprocessor class. The recommend way is to read a config.yaml file and pass the return value as a parameter
        to this object. The mapping file necessarily requires the following keywords:

            >> "mapping" must be a primary key.
        Withing the mapping dictionary, the following second level keywords are mandatory:
            >> "imputer"
            >> "encoder"
            >> "scaler"
            >> "subset_selector"
            >> "categorical_encoder"

        The values obtained by the second level keywords are pointer to the corresponding preprocessor class.

            Example of a yaml mapping:

            >> mapping:
            >>     scaler:
            >>           'minmax': MinMaxScaler
            >>             'standard': StandardScaler
            >>     encoder:
            >>             'onehot': OneHotEncoder
            >>             'label': LabelEncoder
            >>             'binarize': Binarizer
            >>     categorical_encoder:
            >>             'onehot': OneHotEncoder
            >>             'label': LabelEncoder
            >>             'binarize': Binarizer
            >>     subset_selector:
            >>             'name' : NameSelector
            >>             'type' : TypeSelector
            >>     imputer:
            >>             simple : SimpleImputer
            >>             knn : KNNImputer


        :param data_source: provides a string to a file to be read or a plain pandas DataFrame
        :param target_name: the label of the target to be analyzed
        :param config: a config file containing mappings for preprocessor classes
        :param features: optionally, the feature of interest can be defined here
        # TODO make keys optionals
        """
        super().__init__(
            data_source=data_source, target_name=target_name, features=features
        )
        self.features = features
        self._Config = config
        # The General mapping provided by the "mapping" key
        self._Mappings = self.get_config_classes_by_key("mapping")
        # Standard Initialization of an Imputer
        self._Imputer = self._Mappings["imputer"]["simple"](
            **{"missing_values": np.nan, "strategy": "mean"}
        )
        # Standard Initialization of an Encoder
        self._Encoder = self._Mappings["encoder"]["label"]()
        # Standard Initialization of a Scaler
        self._Scaler = self._Mappings["scaler"]["minmax"]()
        # Standard Initialization of a Subset Selector
        self._SubSetSelector = self._Mappings["subset_selector"]["type"]
        # Standard Initialization of a Categorical Encoder
        self._Categorical_Encoder = self._Mappings["categorical_encoder"]["binarize"]()

    @property
    def SubSetSelector(self):
        """
        A property which returns the pandas DataFrame subset selector

        :return: the SubSetSelector attribute
        """
        return self._SubSetSelector

    @SubSetSelector.setter
    def SubSetSelector(self, value):
        """
        A setter method use to reset the SubSet Selector property. It does so by accessing the ``subset_selector`` key of the Mapping configuration
        The value to reset must be a dictionary key in the ``subset_selector`` dictionary

            Example:
            Given the following configuration file entries:

            >> mapping:
            >>    subset_selector:
            >>            'name' : NameSelector
            >>            'type' : TypeSelector

            and an DataPreProcessor object:
            >> preprocessor = DataPreProcessor(data)
            >> preprocessor.ingest()

            The SubSetSelector property can be set using:

            >> preprocessor.SubSetSelector = "name"  # sets the ``NameSelector`` as SubSetSelector
            >> preprocessor.SubSetSelector = "type"  # sets the ``TypeSelector`` as SubSetSelector

        :param value: a dictionary mapping key which points to SubSetSelector class
        :return: the object whose SubSetSelector property was reset
        """
        self._SubSetSelector = self._Mappings["subset_selector"][value]

    @property
    def Scaler(self):
        """
        A property which returns a Scaler instance for numerical data

        :return: the Scaler attribute
        """
        return self._Scaler

    @Scaler.setter
    def Scaler(self, value):
        """
        A setter method use to reset the Scaler property. It does so by accessing the ``scaler`` key of the Mapping configuration
        The value to reset must be a dictionary key in the ``scaler`` dictionary.

        Note that this property either receives a single string which is found in the ``scaler`` dictionary or receives a tuple
        where the first entry is a plain string which is found in the ``scaler`` dictionary and the second entry are class's options
        using a dictionary with function arguments as a string as keys and the option value as value. If the input is a single string, standard
        initialization parameters are used.

            Example:
            Given the following configuration file entries:

            >> mapping:
            >>    scaler:
            >>            'minmax' : MinMaxScaler
            >>            'standard' : StandardScaler

            and an DataPreProcessor object:
            >> preprocessor = DataPreProcessor(data)
            >> preprocessor.ingest()

            The Scaler property can be set using:

            >> preprocessor.Scaler = "standard"  # sets the ``StandardScaler`` as Scaler

            Use a tuple to set options within initialization. A StandardScaler with the option ``with_mean`` as False:

            >> preprocessor.Scaler = "standard" , {"with_mean": False} # sets the ``TypeSelector`` as SubSetSelector

        :param value: a dictionary mapping key which points to a Scaler class
        :return: the object whose Scaler property was reset
        """
        if isinstance(value, str):
            log.info(f"Setting Scaler to {value} with standard arguments")
            self._Scaler = self._Mappings["scaler"][value]()
        elif isinstance(value, tuple):
            log.info(f"Setting Scaler to {value[0]} with arguments {value[1]}")
            self._Scaler = self._Mappings["scaler"][value[0]](**value[1])
        else:
            raise NotImplementedError(
                "The Scaler property setter only accepts a string or a tuple as input"
            )

    @property
    def Imputer(self):
        """
        A property which returns a Imputer instance which replaces missing values.

        :return: the Imputer attribute
        """
        return self._Imputer

    @Imputer.setter
    def Imputer(self, value):
        """
        A setter method use to reset the Imputer property. It does so by accessing the ``imputer`` key of the Mapping configuration
        The value to reset must be a dictionary key in the ``imputer`` dictionary.

        Note that this property either receives a single string which is found in the ``imputer`` dictionary or receives a tuple
        where the first entry is a plain string which is found in the ``imputer`` dictionary and the second entry are class's options
        using a dictionary with function arguments as a string as keys and the option value as value. If the input is a single string, standard
        initialization parameters are used.

            Example:
            Given the following configuration file entries:

            >> mapping:
            >>      imputer:
            >>            'simple' : SimpleImputer
            >>            'knn' : KNNImputer

            and an DataPreProcessor object:
            >> preprocessor = DataPreProcessor(data)
            >> preprocessor.ingest()

            The Imputer property can be set using:

            >> preprocessor.Imputer = "knn"  # sets the ``KNNImputer`` as Imputer

            Use a tuple to set options within initialization. A KNNImputer with the option ``n_neighbors`` as 12:

            >> preprocessor.Imputer = "knn" , {"n_neighbors": 12} # sets the ``KNNImputer`` as Imputer

        :param value: a dictionary mapping key which points to a Imputer class
        :return: the object whose Imputer property was reset
        """
        if isinstance(value, str):
            log.info(f"Setting Imputer to {value} with standard arguments")
            self._Imputer = self._Mappings["imputer"][value]()
        elif isinstance(value, tuple):
            log.info(f"Setting Imputer to {value[0]} with arguments {value[1]}")
            self._Imputer = self._Mappings["imputer"][value[0]](**value[1])
        else:
            raise NotImplementedError(
                "The Imputer property setter only accepts a string or a tuple as input"
            )

    @property
    def Encoder(self):
        """
        A property which returns a Encoder instance which encodes string values

        :return: the Encoder attribute
        """
        return self._Encoder

    @Encoder.setter
    def Encoder(self, value):
        """
        A setter method use to reset the Encoder property. It does so by accessing the ``encoder`` key of the Mapping configuration
        The value to reset must be a dictionary key in the ``encoder`` dictionary.

        Note that this property either receives a single string which is found in the ``encoder`` dictionary or receives a tuple
        where the first entry is a plain string which is found in the ``encoder`` dictionary and the second entry are class's options
        using a dictionary with function arguments as a string as keys and the option value as value. If the input is a single string, standard
        initialization parameters are used.

            Example:
            Given the following configuration file entries:

            >> mapping:
            >>      encoder:
            >>            'onehot': OneHotEncoder
            >>            'label': LabelEncoder
            >>            'binarize': Binarizer

            and an DataPreProcessor object:
            >> preprocessor = DataPreProcessor(data)
            >> preprocessor.ingest()

            The Encoder property can be set using:

            >> preprocessor.Encoder = "binarize"  # sets the ``Binarizer`` as Encoder

            Use a tuple to set options within initialization. A Binarizer with the option ``threshold`` as 0.1:

            >> preprocessor.Encoder = "binarize" , {"threshold": 0.1} # sets the ``Binarizer`` as Encoder

        :param value: a dictionary mapping key which points to an Encoder class
        :return: the object whose Encoder property was reset
        """
        if isinstance(value, str):
            log.info(f"Setting String Encoder to {value} with standard arguments")
            self._Encoder = self._Mappings["encoder"][value]()
        elif isinstance(value, tuple):
            log.info(f"Setting String Encoder to {value[0]} with arguments {value[1]}")
            self._Encoder = self._Mappings["encoder"][value[0]](**value[1])
        else:
            raise NotImplementedError("A function to set the imp")

    @property
    def Categorical_Encoder(self):
        """
        A property which returns a Categorical_Encoder instance which encodes categorical values

        :return: the Categorical_Encoder attribute
        """
        return self._Categorical_Encoder

    @Categorical_Encoder.setter
    def Categorical_Encoder(self, value):
        """
         A setter method use to reset the Categorical_Encoder property. It does so by accessing the ``categorical_encoder`` key of the Mapping configuration
        The value to reset must be a dictionary key in the ``categorical_encoder`` dictionary.

        Note that this property either receives a single string which is found in the ``categorical_encoder`` dictionary or receives a tuple
        where the first entry is a plain string which is found in the ``categorical_encoder`` dictionary and the second entry are class's options
        using a dictionary with function arguments as a string as keys and the option value as value. If the input is a single string, standard
        initialization parameters are used.

            Example:
            Given the following configuration file entries:

            >> mapping:
            >>      categorical_encoder:
            >>            'onehot': OneHotEncoder
            >>            'label': LabelEncoder
            >>            'binarize': Binarizer

            and an DataPreProcessor object:
            >> preprocessor = DataPreProcessor(data)
            >> preprocessor.ingest()

            The Encoder property can be set using:

            >> preprocessor.Categorical_Encoder = "binarize"  # sets the ``Binarizer`` as Categorical_Encoder

            Use a tuple to set options within initialization. A Binarizer with the option ``threshold`` as 0.1:

            >> preprocessor.Categorical_Encoder = "binarize" , {"threshold": 0.1} # sets the ``Binarizer`` as Categorical_Encoder

        :param value: a dictionary mapping key which points to an Categorical_Encoder class
        :return: the object whose Categorical_Encoder property was reset
        """
        if isinstance(value, str):
            log.info(f"Setting Categorical Encoder to {value} with standard arguments")
            self._Categorical_Encoder = self._Mappings["categorical_encoder"][value]()
        elif isinstance(value, tuple):
            log.info(
                f"Setting Categorical Encoder to {value[0]} with arguments {value[1]}"
            )
            self._Categorical_Encoder = self._Mappings["categorical_encoder"][value[0]](
                **value[1]
            )
        else:
            raise NotImplementedError("A function to set the imp")

    def get_config_classes_by_key(self, key):
        """
        Takes a dictionary keys that points to Preprocessor classes as a raw string and converts the raw string to the
        corresponding Python classes

        :param key: a dictionary key which points to a PreProcessor classes as a raw string
        :type key: str
        :return: the converted dictionary
        :rtype: dict
        """
        return self.convert_nested_dictionary_class(self._Config[key])

    def create_pipeline(self):
        """
        Creates a data pre processing pipelines. For each data type, assembles the different data pre processing steps in a list.
        It then checks, which data type is non empty. If it is non-empty the pipeline step is added to the processing pipeline.
        It then returns the fully assembled pipeline.

        The current version of this function operates on the TypeSelector
        # TODO add a version for naming based pre processors

        :return: a pipeline which can be used to fit_transform the original data
        :rtype: Pipeline
        """
        log.info("Generating a data processing pipeline")
        pipe_num = (
            "Numerical",
            Pipeline(
                [
                    ("selector", self.SubSetSelector(self.dtype_num)),
                    ("imputer", self.Imputer),
                    ("scaler", self.Scaler),
                ]
            ),
        )
        pipe_str = (
            "Strings",
            Pipeline(
                [
                    ("selector", self.SubSetSelector(self.dtype_str)),
                    ("encoder", self.Encoder),
                ]
            ),
        )
        pipe_bool = (
            "Boolean",
            Pipeline(
                [
                    ("selector", self.SubSetSelector(self.dtype_bool)),
                    ("transformer", BooleanTransformer()),
                ]
            ),
        )
        pipe_cat = (
            "Categorical",
            Pipeline(
                [
                    ("selector", self.SubSetSelector(self.dtype_cat)),
                    ("categorical_encoder", self.Categorical_Encoder),
                ]
            ),
        )
        # A dictionary used to check which of the different columns have an empty feature list and can therefore be excluded from the pipeline
        type_feature_mapping = {
            "num": (self.numerical_features, pipe_num),
            "str": (self.string_features, pipe_str),
            "bool": (self.bool_features, pipe_bool),
            "cat": (self.categorical_features, pipe_cat),
        }
        pipeline = Pipeline(
            [
                (
                    "feature_transformation",
                    FeatureUnion(
                        n_jobs=-1,
                        transformer_list=[
                            vals[1]
                            for key, vals in type_feature_mapping.items()
                            if vals[0]
                        ],
                    ),
                )
            ]
        )
        return pipeline

    def get_train_test_data(self, as_df=False, test_size=0.2, **options):
        """
        Creates test and train data set based on the ``create_pipeline`` method.

        :param as_df: a flag indicating if the returned data is a pandas DataFrame
        :type as_df: bool
        :param test_size: the proportion of test size in the slitted data
        :type test_size: float
        :param options: Further options of the split_train_test function
        :return: a namedtuple with a ``test`` attribute containing the test data and a ``train`` attribute containing the training data
        :rtype: namedtuple
        """
        pipe = self.create_pipeline()
        log.info("Transforming initial input data")
        data = pipe.fit_transform(self.input)
        log.info("Splitting to train and test data")
        # TODO stratify --> stratify=self.target,
        train, test = train_test_split(
            data, test_size=test_size,  **options
        )
        if as_df:  #
            train, test = pd.DataFrame(train), pd.DataFrame(test)
        data = namedtuple("data", ["train", "test"])
        return data(train, test)

    # TODO add get train data by column transformer
    # TODO add method to preserve feature names


class BooleanTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms all boolean features to a numeric data type (1/0)
    """

    def fit(self, X, y=None, **kwargs):
        return self

    @staticmethod
    def transform(X: int):
        """
        Converts all boolean features by replacing ``True`` values by ``1`` and ``False`` by 0.
        :param X: the Initial DataFrame
        :return:
        """
        return  X * 1