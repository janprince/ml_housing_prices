import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

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

# fill in missing values with median of corresponding features
housing_num = housing.drop("ocean_proximity", axis=1)       # drops non-numerical feature

imputer = SimpleImputer(strategy="median")

# fit imputer to housing_num
imputer.fit(housing_num)

# transform housing_num
X = imputer.transform(housing_num)             # fills null fields with estimated median value of corresponding feature

housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# Integer encoding of categorical and/or text features (representing categories with integers)
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(housing_cat_encoded)