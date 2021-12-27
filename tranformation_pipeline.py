from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion

"""
 Many data transformation steps need to be executed in the right order like
    - Imputer: to handle null values in dataset
    - LabelEncoder: for numerical encoding of categorical/text features
    - StandardScaler: for feature scaling
    - and Combining features.

 Scikit learn provides the Pipeline class to help with such sequences of transformations.
"""

num_attribs = list()

