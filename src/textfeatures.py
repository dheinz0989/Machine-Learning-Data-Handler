from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


#TODO Implements methods for generating text features
class TextFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Derives Features build on Texts in strings analysis
    """

    def __init__(self, **options):
        # Capture the dtype to be selected
        self.options = options

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X):
        # options here
        assert isinstance(X, pd.DataFrame)
        return X