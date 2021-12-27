
"""
Scikit learn provides many useful transformers, but we will need to write our own for custom cleanup operations
or combining specific attributes.
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


#   custom transformer for combining specific features to enable better learning
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


"""
selector transformer: it simply transforms the data by selecting the desired features (numerical and categorical) 
dropping the rest, and converting the resulting DataFrame to a NumPy array.
"""
class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Given a list of attribute names, it transforms the data by selecting all values of the attributes
    and dropping the rest.
    """
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attribute_names].values