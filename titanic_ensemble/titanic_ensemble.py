
    # -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 00:24:49 2018

@author: phucnm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

sns.set(style='white', context='notebook', palette='deep')

# Load data
##### Load train and Test set
import os
os.chdir("C:\\Users\\phucn\\Documents\\git\\kaggle")
train = pd.read_csv(".\\data\\train.csv")
test = pd.read_csv("./data/test.csv")
IDtest = test["PassengerId"]

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
train.loc[Outliers_to_drop] # Show the outliers rows
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
train_len = len(train)
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
dataset = dataset.fillna(np.nan)
dataset.isnull().sum()

train.info()
train.isnull().sum()

train.describe()
# feature analysis
g = sns.heatmap(train[["Survived", "SibSp", "Parch", "Age", "Fare"]].corr(), annot = True, fmt = ".2f", cmap = "coolwarm")
# Explore SibSp feature vs Survived

g = sns.factorplot(x = "SibSp", y = "Survived", data = train, kind = 'bar', size= 6, palette = "muted")
g.despine(left = True)
g = g.set_ylabels("survival probability")

# Explore Parch feature vs Survived
g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 6 , palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
plt.show()
plt.figure()

# age vs survivability
g = sns.FacetGrid(train, col = 'Survived')
g = g.map(sns.distplot, 'Age')
# Explore Age vs Survived
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")

g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color = "Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], color = "Blue", shade = True)

g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived", "Survived"])

#Fare
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
g = sns.distplot(dataset["Fare"], color = "m", label = "Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc = "best")

# apply log to reduce skewness
dataset["Fare"] = dataset["Fare"].map(lambda i:np.log(i) if i > 0 else 0)
g = sns.distplot(dataset["Fare"], color = "b", label = "Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc = "best")

# 3.2 Categorical values
#Sex
g = sns.barplot(x="Sex",y="Survived",data=train)
g = g.set_ylabel("Survival Probability")
train[["Sex", "Survived"]].groupby("Sex").mean()

# Passenger Class
g = sns.factorplot(x = "Pclass", y = "Survived", data = train, kind = "bar", size = 6, palette = "muted")
g = g.despine(left = True)
g = g.set_ylabels("Survival Probability")

# Pclass and Sex
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")

# Embarked
dataset["Embarked"].isnull().sum()
dataset["Embarked"] = dataset["Embarked"].fillna("S")

g = sns.factorplot(x = "Embarked", y = 'Survived', data = train, size = 6, kind = 'bar', palette = 'muted')
g.despine(left = True)
g = g.set_ylabels('survivability')

# Explore Pclass vs Embarked 
g = sns.factorplot("Pclass", col="Embarked",  data=train,
                   size=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")

# Fill missing values
# age
g = sns.factorplot(y = "Age", x = "Sex" , data = dataset, kind = 'box')
g = sns.factorplot(y = "Age", x = "Sex" , hue = "Pclass", data = dataset, kind = 'box')
g = sns.factorplot(y = "Age", x = "Parch", data = dataset, kind = "box")
g = sns.factorplot(y = "Age", x = "SibSp", data = dataset, kind = "box")

dataset["Sex"] = dataset["Sex"].map({"male": 0, "female": 1})
g = sns.heatmap(dataset[["Age","Sex", "SibSp", "Parch", "Pclass"]].corr(), cmap = "BrBG", annot = True)

index_nan_age = list(dataset["Age"][dataset["Age"].isnull()].index)
for i in index_nan_age:
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset["SibSp"] == dataset.iloc[i]["SibSp"]) & 
                      (dataset["Parch"] == dataset.iloc[i]["Parch"]) & 
                      (dataset["Pclass"] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred):
        dataset["Age"].iloc[i] = age_pred
    else:
        dataset["Age"].iloc[i] = age_med
        
g = sns.factorplot(x = "Survived", y = "Age", data = train, kind = "box")
g = sns.factorplot(x = "Survived", y = "Age", data = train, kind = 'violin')

# engine feature
# name

# title in name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].head()
g = sns.countplot(x = "Title", data = dataset)
g = plt.setp(g.get_xticklabels(), rotation = 45)
# convert title to numerical
dataset["Title"] = dataset["Title"].replace(["Lady", "the Countess", "Countess", "Capt", "Col", "Don", "Major", "Dr", "Rev" , "Sir", "Jonkheer", "Dona"], "Rare")
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss": 1, "Ms":1, "Mrs": 1, "Mme":1, "Mlle": 1, "Mr": 2, "Rare": 3})

dataset[dataset["Title"].isnull()]["Name"]

g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master", "Mrs/Ms/Miss/Mme/Mlle", "Mr", "Rare"])

g = sns.factorplot(x = "Title", y = "Survived", data = dataset , kind = "bar")
g = g.set_xticklabels(["Master", "Mrs-Ms", "Mr", "Rare"])
g = g.set_ylabels("Survivability")
        
dataset.drop(labels = ["Name"], axis = 1, inplace = True)

# larger family is harder to evacuate due to their search of relatives
# create family size
dataset["family_size"] = dataset["SibSp"] + dataset["Parch"] + 1
g = sns.factorplot(x = "family_size", y = "Survived", data = dataset)

dataset["family_single"] = dataset["family_size"].map(lambda s:1 if s == 1 else 0)
dataset["family_small"] = dataset["family_size"].map(lambda s:1 if s == 2 else 0)
dataset["family_medium"] = dataset["family_size"].map(lambda s:1 if 3 <= s <= 4 else 0)
dataset["family_large"] = dataset["family_size"].map(lambda s:1 if s >= 5 else 0)

# title , embarked
dataset = pd.get_dummies(dataset, columns = ["Title"])
dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix = "Em")

# cabin
dataset['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])
g = sns.countplot(dataset['Cabin'], order = ['A', 'B', 'C', 'D', 'E', 'F', 'G' , 'T' , 'X'])
g = sns.factorplot(y = 'Survived', x = 'Cabin', kind = 'bar', data = dataset, order = ['A', 'B', 'C', 'D', 'E', 'F', 'G' , 'T' , 'X'])
dataset = pd.get_dummies(dataset, columns = ['Cabin'], prefix = 'Cabin')
dataset.columns
# ticket
dataset['Ticket']
Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit():
        Ticket.append(i.replace(".", "").replace("/", "").strip().split(' ')[0])
    else:
        Ticket.append("X")
        
dataset["Ticket"] = Ticket
dataset["Ticket"].head()
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix = 'T') 
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"], prefix = 'Pc')

dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)


# model
## prepare train test set
train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels = ["Survived"], axis = 1, inplace = True)
train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train.drop(labels = ["Survived"], axis = 1)
#I compared 10 popular classifiers and evaluate the mean accuracy of each of them by a stratified kfold cross validation procedure.
#
#SVC
#Decision Tree
#AdaBoost
#Random Forest
#Extra Trees
#Gradient Boosting
#Multiple layer perceprton (neural network)
#KNN
#Logistic regression
#Linear Discriminant Analysis


kfold = StratifiedKFold(n_splits = 10)
random_state = 2
classifiers = []
classifiers.append(SVC(random_state = random_state))
classifiers.append(DecisionTreeClassifier(random_state = random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state = random_state), random_state = random_state, learning_rate = 0.1))
classifiers.append(RandomForestClassifier(random_state = random_state))
classifiers.append(ExtraTreesClassifier(random_state = random_state))
classifiers.append(GradientBoostingClassifier(random_state = random_state))
classifiers.append(MLPClassifier(random_state = random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []

for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold))
# REMOVE n_jobs parameter lift off error of parallel
# of course it will not be as fast


cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans": cv_means, "CrossValerrors": cv_std, "Algorithm":["SVC","DecisionTree","AdaBoost", "RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})
g= sns.barplot("CrossValMeans", "Algorithm", data = cv_res, palette = "Set3", orient = "h", **{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")

# hyper params tuning

## adaboost
dtc = DecisionTreeClassifier()
ada_dtc = AdaBoostClassifier(dtc, random_state = 7)
ada_param_grid = {"base_estimator__criterion": ["gini", "entropy"],
                  "base_estimator__splitter": ["best", "random"],
                  "algorithm": ["SAMME", "SAMME.R"]}
gs_ada_dtc = GridSearchCV(ada_dtc, param_grid = ada_param_grid, cv = kfold,
                          scoring = 'accuracy', verbose = 1)
gs_ada_dtc.fit(X_train, Y_train)
ada_best = gs_ada_dtc.best_estimator_
gs_ada_dtc.best_score_

## ExtraTrees
extc = ExtraTreesClassifier()
ex_param_grid = {"max_depth": [None],
                 "max_features":[1,3,10],
                 "min_samples_split":[2,3,10],
                 "min_samples_leaf":[1,3,10],
                 "bootstrap":[False],
                 "n_estimators":[100,300],
                 "criterion":["gini"]}

gs_extc = GridSearchCV(extc, param_grid = ex_param_grid, cv = kfold, scoring = 'accuracy', verbose = 1)
gs_extc.fit(X_train, Y_train)
extc_best = gs_extc.best_estimator_
gs_extc.best_score_

## RFC 
rfc = RandomForestClassifier()
rf_param_grid = {"max_depth":[None],
                 "max_features":[1,3,10],
                 "min_samples_split":[2,3,10],
                 "min_samples_leaf":[1,3,10],
                 "bootstrap":[False],
                 "n_estimators":[100,300],
                 "criterion":['gini'],
                 }
gs_rfc = GridSearchCV(rfc, param_grid = rf_param_grid, cv= kfold, scoring = 'accuracy', verbose = 1)
gs_rfc.fit(X_train, Y_train)
rfc_best = gs_rfc.best_estimator_
gs_rfc.best_score_

## Gradient Boosting
gbc = GradientBoostingClassifier()
gb_param_grid = {'loss':["deviance"],
                 'n_estimators':[100, 200, 300],
                 'learning_rate':[0.1, 0.05, 0.01],
                 'max_depth': [4,8],
                 'min_samples_leaf':[100, 500],
                 'max_features':[0.3, 0.1]}
gs_gbc = GridSearchCV(gbc, param_grid = gb_param_grid, cv = kfold, scoring = 'accuracy', verbose = 1)
gs_gbc.fit(X_train,Y_train)
gbc_best = gs_gbc.best_estimator_
gs_gbc.best_score_

## support vector machine
SVMC = SVC(probability = True)
svc_param_grid = {'kernel':['rbf'],
                  'gamma':[0.001,0.01,0.1,1],
                  'C':[1,10,50,100,200,300,1000]}
gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", verbose = 1)

gsSVMC.fit(X_train,Y_train)

svc_best = gsSVMC.best_estimator_
gsSVMC.best_score_


# Learning curve ---------
def plot_learning_curve(estimator, title, X, y, ylim = None, cv = None,
                        train_sizes = np.linspace(0.1,1,5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training examples')
    plt.ylabel('score')
    train_sizes, train_scores, test_scores = learning_curve(estimator, X,y,cv = cv, train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_score_mean = np.mean(test_scores, axis = 1)
    test_score_td = np.std(test_scores, axis = 1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                     train_scores_mean + train_scores_std , alpha = 0.1,
                     color = 'r')
    plt.fill_between(train_sizes, test_score_mean - test_score_td, 
                     test_score_mean + test_score_td , alpha = 0.1,
                     color = 'g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color = 'r', label = 'Training score')
    plt.plot(train_sizes, test_score_mean, 'o-', color = 'g', label = 'Cross-validation score')
    plt.legend(loc = 'best')
    return(plt)
    
g = plot_learning_curve(rfc_best, 'rf learning curve', X_train, Y_train, cv = kfold)
g = plot_learning_curve(ada_best, 'adaboost learning curve', X_train, Y_train, cv = kfold)
g = plot_learning_curve(gbc_best, 'gradient boosting learning curve', X_train, Y_train, cv = kfold)
g = plot_learning_curve(svc_best, 'support vector machine learning curve', X_train, Y_train, cv = kfold)
g = plot_learning_curve(extc_best, 'extra trees learning curve', X_train, Y_train, cv = kfold)
# feature important
nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex = 'all', figsize  = (15,15))
names_classifiers = [("AdaBoosting", ada_best), ("ExtraTrees", extc_best), ("RandomForest", rfc_best), ("GradientBoosting", gbc_best)]

n_classifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[n_classifier][0]
        classifier = names_classifiers[n_classifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y = X_train.columns[indices][:40], x = classifier.feature_importances_[indices][:40], orient = 'h', ax = axes[row][col])
        g.set_xlabel("relative important", fontsize = 12)
        g.set_ylabel('features', fontsize = 12)
        g.tick_params(labelsize = 9)
        g.set_title(name + " feature importance")
        n_classifier += 1
        # set chart area by 2x2, each moving forward one type of estimator
Y_train
X_train.columns
# test
test_rfc = pd.Series(rfc_best.predict(test), name = 'RFC')
test_ada = pd.Series(ada_best.predict(test), name = 'AdaBoost')
test_gbc = pd.Series(gbc_best.predict(test), name = 'GradientBoosting')
test_svc = pd.Series(svc_best.predict(test), name = 'SupportVectorMachine')
test_extc = pd.Series(extc_best.predict(test), name = 'ExtraTrees')
ensemble_results = pd.concat([test_rfc, test_ada, test_gbc, test_svc, test_extc], axis = 1)

g = sns.heatmap(ensemble_results.corr(), annot = True)

# ensemble voting ---------
vote_count = VotingClassifier(estimators = [('rfc', rfc_best),
                                            ('ada', ada_best),
                                            ('gbc', gbc_best),
                                            ('svc', svc_best),
                                            ('extc', extc_best)])
    
vote_count = vote_count.fit(X_train, Y_train)

# predict ---------
test_survived = pd.Series(vote_count.predict(test), name = 'Survived')
results = pd.concat([IDtest,  test_survived], axis = 1)
results.to_csv('ensemble_python_voting.csv', index = False)
