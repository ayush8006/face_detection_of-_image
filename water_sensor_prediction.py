# -*- coding: utf-8 -*-
"""Water Sensor Prediction

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PXhkx1vEictVMTFHAdhGZj-wJ9QXhHh9
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/ektanegi25/Water-Sensor-repository/main/wafer_23012020_041211.csv")
df.head()

df.tail()

df.rename(columns = {'Unnamed: 0':"Wafers"}, inplace = True)

df

from sklearn.model_selection import train_test_split


df, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
df.info()

df_test.info()

df.describe()

df['Good/Bad'].isnull().sum()

df.isnull().sum()

df.isnull().sum().sum()

df.isnull().sum().sum()/ (df.shape[0] * (df.shape[1] - 1)) *100

df_test.isnull().sum().sum()

df

# let's have a look at the distribution first 50 sensors of Wafers

plt.figure(figsize=(15, 100))

for i, col in enumerate(df.columns[1:51]):
    plt.subplot(60, 3, i+1)
    sns.histplot(x=df[col], color='indianred')
    plt.xlabel(col, weight='bold')
    plt.tight_layout()

random_50_sensors = []
for i in range(50):
    if i not in random_50_sensors:
        random_50_sensors.append(np.random.randint(1,591))

# let's have a look at the distribution first 50 sensors of Wafers

plt.figure(figsize=(15, 100))

for i, col in enumerate(df.columns[random_50_sensors]):
    plt.subplot(60, 3, i+1)
    sns.histplot(x=df[col], color='indianred')
    plt.xlabel(col, weight='bold')
    plt.tight_layout()

def get_cols_zero_std(df : pd.DataFrame):
    cols_to_drop = []
    num_cols = [i for i in df.columns[1:] if df[i].dtype != "O"]
    for i in df.columns[1:]:
        if df[i].std() ==0:
            cols_to_drop.append(i)
    return cols_to_drop

def get_reduntant_col(df: pd.DataFrame, missing_thresh = .7):
    cols_missing_ratio = df.isnull().sum().div(df.shape[0])
    cols_to_drop = list(cols_missing_ratio[cols_missing_ratio > missing_thresh].index)
    return cols_to_drop

df

cols_drop_1 = get_cols_zero_std(df = df)
cols_drop_1

cols_drop_2 = get_reduntant_col(df, missing_thresh= .7)
cols_drop_2

cols_to_drop = cols_drop_1 +cols_drop_2 +['Wafers']
len(cols_to_drop)

df

X, y = df.drop(cols_to_drop, axis = 1), df['Good/Bad']

X.shape

y.shape

y.value_counts()

from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler

imputer = KNNImputer(n_neighbors= 3)
prep_process = Pipeline(steps = [('Imputer', imputer), ('Scaling' ,RobustScaler()) ])
prep_process

X_trans = prep_process.fit_transform(X)
X_trans.shape

# Commented out IPython magic to ensure Python compatibility.
# %pip install kneed

from sklearn.cluster import KMeans
from kneed import KneeLocator
import numpy as np
from typing import Tuple

def cluster_data_instances(X: np.array) -> Tuple[KMeans, np.array]:
    wcss = []  # Within Summation of Squares

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    knee_finder = KneeLocator(
        range(1, 11), wcss, curve='convex', direction='decreasing')
    ideal_clusters = knee_finder.knee

    kmeans = KMeans(n_clusters=ideal_clusters, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    return kmeans, np.c_[X, y_kmeans]

# Usage example:
kmeans, X_clus = cluster_data_instances(X_trans)

kmeans

X_clus

np.unique(X_clus[:,-1])

wafer_clus = np.c_[X_clus, y]
wafer_clus

wafer_clus[wafer_clus[:,-2] == 0].shape

wafer_clus[wafer_clus[:,-2] == 1].shape

wafer_clus[wafer_clus[:,-2] == 2].shape

from imblearn.combine import SMOTETomek

X, y = X_trans[:,:-1], y
resampler = SMOTETomek(sampling_strategy= "auto")
X_res, y_res = resampler.fit_resample(X,y)

print("Before resampling, Shape of training instances: ", np.c_[X, y].shape)
print("After resampling, Shape of training instances: ", np.c_[X_res, y_res].shape)

## Target Cats after Resampling

print(np.unique(y_res))
print(f"Value Counts: \n-1: {len(y_res[y_res == -1])}, 1: {len(y_res[y_res == 1])}")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=1/3, random_state=42)

print(f"train set: {X_train.shape, y_train.shape}")
print(f"test set: {X_test.shape, y_test.shape}")

# Prepared training and test sets

X_prep = X_train
y_prep = y_train
X_test_prep = X_test
y_test_prep = y_test

print(X_prep.shape, y_prep.shape)
print(X_test_prep.shape, y_test_prep.shape)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score


svc_clf = SVC(kernel='linear')
svc_rbf_clf = SVC(kernel='rbf')
random_clf = RandomForestClassifier(random_state=42)
xgb_clf = XGBClassifier()

## A function to display Scores

def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

## SVC Scores

svc_scores = cross_val_score(svc_clf, X_prep, y_prep, scoring='roc_auc', cv=10, verbose=2)
display_scores(svc_scores)

## Performance on test set using cross-validation

# Predictions using cross-validation
svc_preds = cross_val_predict(svc_clf, X_test_prep, y_test_prep, cv=5)

# AUC score
svc_auc = roc_auc_score(y_test_prep, svc_preds)
svc_auc

## SVC rbf Scores

svc_rbf_scores = cross_val_score(svc_rbf_clf, X_prep, y_prep, scoring='roc_auc', cv=10, verbose=2)
display_scores(svc_rbf_scores)

## Performance on test set using cross-validation

# Predictions using cross-validation
svc_rbf_preds = cross_val_predict(svc_rbf_clf, X_test_prep, y_test_prep, cv=5)

# AUC score
svc_rbf_auc = roc_auc_score(y_test_prep, svc_rbf_preds)
svc_rbf_auc

## Random Forest Scores

random_clf_scores = cross_val_score(random_clf, X_prep, y_prep, scoring='roc_auc', cv=10, verbose=2)
display_scores(random_clf_scores)

## Performance on test set using cross-validation

# Predictions using cross-validation
random_clf_preds = cross_val_predict(random_clf, X_test_prep, y_test_prep, cv=5)

# AUC score
random_clf_auc = roc_auc_score(y_test_prep, random_clf_preds)
random_clf_auc

## XGB

xgb_clf_scores = cross_val_score(xgb_clf, X_prep, y_prep, scoring='roc_auc', cv=10, verbose=2)
display_scores(xgb_clf_scores)

y_prep
