import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tranformation_pipeline import pipeline

#  analysis and visualization of data is done in the attached notebook. :)

# load data
housing = pd.read_csv("../dataset/housing.csv")

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


# selecting and training a model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# model 0
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# model 1
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

# model 2
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

# predictions by the two models
# predictions = lin_reg.predict(housing_prepared)
# predictions = tree_reg.predict(housing_prepared)
predictions = forest_reg.predict(housing_prepared)    # best

# some_data = housing.iloc[:5]
# some_labels = housing_labels[:5]
# some_data_prepared = full_pipeline.transform(some_data)
# print(f"Predictions:\t\t {forest_reg.predict(some_data_prepared)}")
# print(f"Original Labels:\t\t {list(some_labels)}")


# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)
#
# def display_scores(scores):
#     print("Scores: ", scores)
#     print("Mean: ", scores.mean())
#     print("Standard Deviation: ", scores.std())
#
# display_scores(rmse_scores)

# evaluate system on test set
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)

final_predictions = forest_reg.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

