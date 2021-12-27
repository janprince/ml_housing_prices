from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
import pandas as pd
from custom_transformers import DataFrameSelector, CombinedAttributesAdder

"""
 Many data transformation steps need to be executed in the right order like
    - Imputer: to handle null values in dataset
    - LabelEncoder: for numerical encoding of categorical/text features
    - StandardScaler: for feature scaling
    - and Combining features.

 Scikit learn provides the Pipeline class to help with such sequences of transformations.
"""

def pipeline(data: pd.DataFrame):

    # copy of only numerical attributes
    housing_num = data.drop("ocean_proximity", axis=1)

    # lists of attribute names (Strings)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    # pipeline for handling numerical attributes
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),   # returns a numpy array with values of the attribs in num_attribs
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),               # for feature scaling
    ])

    # pipeline to handle categorical/text attributes
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('one_hot_encoder', OneHotEncoder(sparse=False)),    # numerical encoding of categorical attributes
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)
    ])

    return full_pipeline

"""
FeatureUnion class: You give it a list of transformers (which can be entire transformer pipelines), and when its transform()
method is called it runs each transformer’s transform() method in parallel, waits for their output, 
and then concatenates them and returns the result (and of course calling its fit() method calls all each 
transformer’s fit() method).
"""