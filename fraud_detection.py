import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

data = pd.read_csv('creditcard.csv')

print(data.columns)

print(data.shape)

data = data.sample(frac = 0.1, random_state= 1)

print(data.shape)

#Determing number of fraud cases in a dataset

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud) / float(len(Valid))

print(outlier_fraction)

print("fraud cases: {}".format(len(Fraud)))
print("valid cases: {}".format(len(Valid)))

#correlation matrix
#see if there are any sstrong correlations between variables in dataset
corrmat = data.corr()
fig = plt.figure(figsize= (12, 9))

#format dataset

#Get all columns from dataset
columns = data.columns.tolist()

#remove unnecessary columns (remove 'class' column0
columns = [c for c in columns if c not in ["Class"]]

#store the variable we'll be predicting on
target = "Class"

X = data[columns]
#Target means this is what we are trying to determine
#Class: 0 = valid, 1 = fraudulent
y = data[target]

#Print the shapes of X and y
print(X.shape)
print(y.shape)

#Start building our networks using isolation forest algorithm and local outlier factor algorithm
#Try and do anomaly detection on dataset

#49 anomaly cases (frauds)

#define a random state
state = 1

#define the outlier detection methods
#dictionary of classifiers:
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors= 20,
        contamination= outlier_fraction
    )
}

#fit the model

#number of outliers
n_outliers = len(Fraud)

#do for loop through 2 different classifiers defined above
#enumerate list of classifiers so we can cycle through them
#can index dictionary using .item

for i, (clf_name, clf) in enumerate(classifiers.items()):
    #fit the data and log outliers

    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scored_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scored_pred = clf.decision_function(X)
        y_pred = clf.predict(X)


# y prediction values: -1 = outlier, 1 = inlier
# need to process information before comparing to class labels(0 = valid, 1 = fraud)
# index y prediction:
    # if = 1 -> 0
    # if = -1 -> 1

y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

# Calculate number of errors -> use comparison to y (target)
# Total number of errors:
n_errors = (y_pred != y).sum()

#Run classification metrics to get useful information
print('{}. {}'.format(clf_name, n_errors))
print(accuracy_score(y, y_pred))
print(classification_report(y, y_pred))







