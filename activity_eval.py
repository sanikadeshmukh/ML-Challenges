"""
Sanika Prashant Deshmukh, date - 04-23-25

Assignment 2 - Give Your Models a Grade - AI 541, Spring 2025

This script trains and evaluates three classifiers (Baseline, RandomForest, KNeighbors, and GaussianNB)
on the original development dataset (activity-dev.csv) using five evaluation methodologies:

1. Train-test split (80% train, 20% test)
2. 10-fold cross-validation
3. Stratified 10-fold cross-validation
4. Group-wise 10-fold cross-validation
5. Stratified group-wise 10-fold cross-validation

It then evaluates each trained model on a separate held-out dataset (activity-heldout.csv)
and compares those results to the generalization estimates from the development data.
"""
# importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the activity-dev dataset
data = pd.read_csv("activity-dev.csv")

# Load the activity-heldout dataset
blind_data = pd.read_csv("activity-heldout.csv")

# Drop activity and person columns from data
# put group column in a separate variable for group related cross-validation
X = data.drop(columns=["activity", "person"])
y = data["activity"]
groups = data["person"]

#Baseline - DummyClassifier
baseline = DummyClassifier(
    strategy="stratified",
    random_state=42)

# RandomForestClassifier - RF
rf = RandomForestClassifier(
    n_estimators=100,       
    min_samples_leaf=5,    
    random_state=42         
)

# KNeighborsClassifier - K-NN 
knn = KNeighborsClassifier(
    n_neighbors=5,          #no of neighbors
    weights="uniform",      #uiform weights
    metric="minkowski",     #distance metric
    p=2                     #p=2 for euclidean
)

# GaussianNB - NB
gnb = GaussianNB(var_smoothing=1e-9)      #smoothing parameter     


# Dict of classifiers                    
classifiers = {
    "Baseline": baseline,
    "RF": rf,
    "K-NN": knn,
    "NB": gnb
}

# Function takes in classifiers, training and testing data, and returns accuracy scores
def evaluate_models(classifiers, X_train, X_test, y_train, y_test):
    acc_scores = {}
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        acc_scores[name] = acc
    return acc_scores
    
cv_result = {} # Store results

# attribution - cross-validation methodologies and their usage patterns (e.g., KFold, 
# StratifiedKFold, GroupKFold, StratifiedGroupKFold) are taken as eg from 
# scikit-learn's official documentation and examples

#---------------------- 80% train, 20% test split)-------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)
cv_result["A:Train 80% test 20%"] = evaluate_models(classifiers, X_train, X_test, y_train, y_test)   

#---------------------- 10 fold cross-validation ------------------------
cv_result["B:10 fold CV"] = {name: [] for name in classifiers}
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# for each fold, train and test the model and store the accuracy scores
for train_idx, test_idx in kfold.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    fold_scores = evaluate_models(classifiers, X_train, X_test, y_train, y_test)
    for name, acc in fold_scores.items():
        cv_result["B:10 fold CV"][name].append(acc)

# average of fold scores
cv_result["B:10 fold CV"] = {name: np.mean(accs) for name, accs in cv_result["B:10 fold CV"].items()}

#---------------------- Stratified 10 fold cross-validation ------------------------
cls_stratified = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_result["C:Stratified 10-fold CV"] = {name: [] for name in classifiers}

# for each fold, train and test the model and store the accuracy scores
for train_idx, test_idx in cls_stratified.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    clsstr_scores = evaluate_models(classifiers, X_train, X_test, y_train, y_test)
    for name, acc in clsstr_scores.items():
        cv_result["C:Stratified 10-fold CV"][name].append(acc)

# average of fold scores
cv_result["C:Stratified 10-fold CV"] = {name: np.mean(accs) for name, accs in 
                                       cv_result["C:Stratified 10-fold CV"].items()}

#----------------------Groupwise 10 fold cross-validation ------------------------
grp_kfold  = GroupKFold(n_splits=10, shuffle=True, random_state=42)
cv_result["D:Groupwise 10-fold CV"] = {name: [] for name in classifiers}

# for each fold, train and test the model and store the accuracy scores
for train_idx, test_idx in grp_kfold.split(X, y,groups=groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    group_scores = evaluate_models(classifiers, X_train, X_test, y_train, y_test)
    for name, acc in group_scores.items():
        cv_result["D:Groupwise 10-fold CV"][name].append(acc)

# average of fold scores
cv_result["D:Groupwise 10-fold CV"] = {name: np.mean(accs) for name, accs in 
                                      cv_result["D:Groupwise 10-fold CV"].items()}
#---------------------- Stratified Groupwise 10 fold CV ------------------------
str_grp_kfold  = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
cv_result["E:Stratified groupwise 10-fold CV"] = {name: [] for name in classifiers}

# for each fold, train and test the model and store the accuracy scores
for train_idx, test_idx in str_grp_kfold.split(X, y,groups=groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    strgroup_scores = evaluate_models(classifiers, X_train, X_test, y_train, y_test)
    for name, acc in strgroup_scores.items():
        cv_result["E:Stratified groupwise 10-fold CV"][name].append(acc)

# average of fold scores
cv_result["E:Stratified groupwise 10-fold CV"] = {name: np.mean(accs) for name, accs in 
                                                 cv_result["E:Stratified groupwise 10-fold CV"].items()}

#----------------------end of cross-validation ------------------------

# Store results in a DataFrame and print
df_results = pd.DataFrame(cv_result)
print(df_results.round(4).to_string())


# Evaluate on activity-heldout dataset
X_dev = data.drop(columns=["activity", "person"])
y_dev = data["activity"]
X_heldout = blind_data.drop(columns=["activity", "person"])
y_heldout = blind_data["activity"]
results = {} # store results


# Evaluate the models on heldout data
results["Held out accuracy"] = evaluate_models(classifiers, X_dev, X_heldout, y_dev, y_heldout)
df_results  = pd.DataFrame(results)
print(df_results.round(4).to_string()) #show results for heldout data


error_diff = {} # Store error differences
# Calculate the difference between estimates and actuals

for method in cv_result:
    error_diff[method] = {}
    for model in cv_result[method]:
        est = cv_result[method].get(model)
        actual = results["Held out accuracy"].get(model)
        if est is not None and actual is not None:
            error_diff[method][model] = est - actual  

df_error = pd.DataFrame(error_diff)   # make a dataframe of error differences
df_error.loc["Avg"] = df_error.mean()   # average error differences for each method

# Round the error differences to 4 decimal places
print("\nSigned Error Table (Estimate - Actual):")
print(df_error.round(4).to_string())

# Confusion matrix for best model (GaussianNB)
model = GaussianNB() # best model for held-out evaluation
model.fit(X_dev, y_dev) # train the model on the development data
y_pred = model.predict(X_heldout) # predict on the held-out data

# Confusion matrix for the best model (GaussianNB)
cm = confusion_matrix(y_heldout, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix - Gaussian NB on Held-out Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

