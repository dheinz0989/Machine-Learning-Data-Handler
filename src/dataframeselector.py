"""
This module includes source code to filter Pandas DataFrame to pre-defined columns.
They inherit the BaseEstimator and TransformerMixin classes from scikit-learn which add a ``fit_transform`` method and get a new Data Frame.
There are two DataFrame selector classes:

    - ``TypeSelector``: accepts a data type as input and returns a DataFrame with all columns of the data type
    - ``NameSelector``: in contrast, accepts a string or a list of feature names and returns all column whose name is given

Furthermore, two other classes are inside this class

    - it includes a BooleanTransformer which allows to transform boolean features to numerical
    - it includes a DummyEncoder which can be used to create dummy columns out of Categorical Features

"""
from __future__ import annotations
from typing import Union, Sequence
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from numpy import number


class TypeSelector(BaseEstimator, TransformerMixin):
    """
    Selects a subset of columns from a DataFrame based on the type of data
    """

    def __init__(self, dtype: Union[str, int, float, number, pd.datetime]):
        """
        Initializes a Type Selector object that selects subset of the DataFrame

        :param dtype: the given data type for selection
        :type dtype: Union[str, int, float, number, pd.datetime]
        """
        self.dtype = dtype

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        The method to select the data based on the data type

        :param X: the Initial DataFrame
        :return:
        """
        X = X.to_frame() if isinstance(X, pd.Series) else X
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


class NameSelector(BaseEstimator, TransformerMixin):
    """
    Selects a subset of columns from a DataFrame based on a list of feature name
    """

    def __init__(self, features: Union[str, Sequence[str]]):
        """
        Specify the feature name(s) to be selected. Either as a string of a feature or a list of features

        :param features: the features to be selected
        :type features: Union[str, Sequence[str]]
        """
        self.features = features if isinstance(features, list) else [features]

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        The method to select the data based on feature names

        :param X: the Initial DataFrame
        :return:
        """
        X = X.to_frame() if isinstance(X, pd.Series) else X
        assert all([
            isinstance(X, pd.DataFrame),
            all([ftr in list(X.columns) for ftr in self.features])
        ])
        return X[self.features]



