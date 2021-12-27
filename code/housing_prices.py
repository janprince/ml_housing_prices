import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tranformation_pipeline import pipeline

#  analysis and visualization of data is done in the attached notebook. :)

# load data
housing = pd.read_csv("dataset/housing.csv")

# creating a new feature
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# splitting dataset (Performing stratified sampling based on the income category.)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# drop income_cat feature
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

# separate predictors from labels
housing = strat_train_set.drop("median_house_value", axis=1)    # predictors
housing_labels = strat_train_set["median_house_value"].copy()

# Apply transformation pipeline to data
full_pipeline = pipeline(housing)
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)