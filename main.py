# imports
import pandas as pd
import json
from sklearn.cluster import k_means
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib.pyplot as plt

# load dataset
raw_data = pd.read_csv("data/incident_event_log.csv")

# data preprocessing

# load preprocessing metadata
with open("data/preprocmd.json") as f:
    md = json.load(f)

def map_categories(dataset, col_mappings):
    results = dataset
    for column_name in col_mappings.keys():
        results = results.replace({column_name: col_mappings[column_name]})
    return results

def extract_numval(x):
        match = re.search(r'(\d+)', x)
        if match:
            result = int(match.group())
        else:
            result = int(-1) # default value
        return result

# Transform categorical values to integers
incilog = map_categories(raw_data, md["col_mappings"])
for column_name in md["numval_cols"]:
    incilog[column_name] = list(map(extract_numval, incilog[column_name]))

# Format dates
for column_name in md["datetime_cols"]:
    incilog[column_name] = pd.to_datetime(incilog[column_name].map(lambda x: re.sub(r'\?', '', x)))

# Target variable
incilog["time_to_resolution"] = [td.seconds for td in (incilog["resolved_at"] - incilog["opened_at"])]
# Remove negative deltas (either wrong date entries in the resolved_at column or redundant incidents, already resolved)
incilog = incilog[incilog.time_to_resolution >= 0]

# Data segmentation
train, test = train_test_split(incilog, test_size=0.2, random_state=25)

# classify columns
excluded_cols = ["number"] + md["datetime_cols"]
categorical_cols = incilog.columns.difference(excluded_cols).difference(["time_to_resolution"])


# exploratory analysis

print(incilog.head().to_markdown())

# correlations with time_to_resolution
def res_correlation(variable):
    return incilog[variable].corr(incilog.time_to_resolution)

correlations = pd.Series([res_correlation(col) for col in categorical_cols], index = categorical_cols)
# Highest "single-factor" correlations
top_15 = correlations[correlations.abs().sort_values(ascending=False).index].head(15)
print(top_15.to_markdown())


# plot distribution of categorical variables
def barplot(variable):
    dist = pd.Series(incilog[variable]).value_counts()
    plt.bar(range(len(dist.unique)), dist)

fig1, axes1 = plt.subplots(5, 3, figsize=(12, 10))
fig1.subplots_adjust(wspace=0.4, hspace=0.6)
fig1.suptitle("Figure 1. Frequencies of the descriptive variables most correlated with the time_to_resolution")
for c in range(len(top_15.index)):
    variable = top_15.index[c]
    dist = pd.Series(incilog[variable]).value_counts()
    range_vals = range(len(dist.keys()))
    axes1[divmod(c, 3)[0]][divmod(c, 3)[1]].set_title(variable)
    axes1[divmod(c, 3)[0]][divmod(c, 3)[1]].bar(range_vals, dist)
fig1.savefig("./images/top_15_barplots.png")

# First regressor and feature selection

# random forest
def fit_random_forest(train, features=categorical_cols, n_estimators=50):
    forest_model = RandomForestRegressor(n_estimators=n_estimators, criterion="mse")
    forest_model.fit(X=train.filter(items=features),
                     y=train.time_to_resolution)
    return forest_model

forest_all_features = fit_random_forest(train, features=categorical_cols)
R2_all_features = forest_all_features.score(test.filter(items=categorical_cols), test.time_to_resolution)
print('R2 with all features: {:f}'.format(R2_all_features))

features_importance = categorical_cols[np.argsort(forest_all_features.feature_importances_)[::-1]]
print(pd.Series(features_importance[:15]).to_markdown())

# forest_top_15 = fit_random_forest(train, features=top_15.index)
# R2_top_15_corr = forest_top_15.score(test.filter(items=top_15.index), test.time_to_resolution)
# print('R2 with top 15 correlated features: {:f}'.format(R2_top_15))


# reduce the number of features in the model and evaluate

# first criterion for ranking features: pair-wise correlation with time_to_resolution
# R2_corr = [R2_top_15_corr]
R2_corr = []
for i in range(15, len(categorical_cols)):
    features = correlations.abs().sort_values(ascending=False).index[:i]
    forest = fit_random_forest(train, features)
    R2_corr.append(forest.score(test.filter(items=features), test.time_to_resolution))
R2_corr.append(R2_all_features)


# second criterion for ranking features: features importance of full random forest
R2_imp = []
for i in range(15, len(categorical_cols)):
    features = features_importance[:i]
    forest = fit_random_forest(train, features)
    R2_imp.append(forest.score(test.filter(items=features), test.time_to_resolution))
R2_imp.append(R2_all_features)

R2 = pd.DataFrame({"corr": R2_corr, "imp": R2_imp}, index = range(15, len(categorical_cols)+1))
R2.to_pickle("./R2.pkl")


fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.subplots_adjust(hspace=0.6)
fig2.suptitle("Figure 2. Evolution of the random forest scores")
axes2[0].set_title("Features ranked by correlation")
axes2[0].plot(R2["corr"])
axes2[0].set_xlabel("Number of features")
axes2[0].set_ylabel("$R^2$")
axes2[0].set(ylim=(0.44, 0.65))
axes2[1].set_title("Features ranked by random forest importance")
axes2[1].plot(R2["imp"])
axes2[1].set_xlabel("Number of features")
axes2[1].set_ylabel("$R^2$")
axes2[1].set(ylim=(0.44, 0.65))

plt.savefig("./images/R2.png")


# clustering
def find_clusters(included_cols, max_clusters=11):
    inertiae = {}
    best_n_iters = {}
    clusters = {}
    for c in range(1, max_clusters):
        km = k_means(incilog[included_cols], n_clusters=c, return_n_iter=True)
        inertiae[c] = km[2]
        best_n_iters[c] = km[3]
        clusters[c] = km[1]
    return inertiae, best_n_iters, clusters

# clusters with all categorical columns
inertiae_all, best_n_iters_all, clusters_all = find_clusters(categorical_cols)


fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.subplots_adjust(hspace=0.6)
fig3.suptitle("Figure 3. K-means metrics")
axes3[0].set_title("K-means inertiae")
axes3[0].plot(inertiae_all.keys(), inertiae_all.values(), marker = "o")
axes3[0].set_xlabel("Number of clusters")
axes3[0].set_ylabel("Inertiae")
axes3[1].set_title("Number of iterations corresponding to the best results")
axes3[1].scatter(best_n_iters_all.keys(), best_n_iters_all.values())
axes3[1].set_xlabel("Number of clusters")
axes3[1].set_ylabel("Number of iterations")
plt.savefig("./images/k_means.png")

# Elbow: 4 clusters
incilog["cluster_all_vars"] = clusters_all[4]

# Explain the clusters
# important features on each cluster
def clusters_features_importance(cluster_column):
    cluster_features_importance={}
    for c in range(4):
        forest = fit_random_forest(incilog[incilog[cluster_column] == c])
        cluster_features_importance[c] = categorical_cols[np.argsort(forest.feature_importances_)[::-1]]
    return cluster_features_importance

# not very conclusive

# find clusters using less features
inertiae_10, best_n_iters_10, clusters_10 = find_clusters(features_importance[0:10])
plt.plot(inertiae_10.keys(), inertiae_10.values(), marker = "o")
plt.scatter(best_n_iters_10.keys(), best_n_iters_10.values())
# 4 clusters
incilog["cluster_10_vars"] = clusters_10[4]
cf_10_var = clusters_features_importance("cluster_10_vars")
