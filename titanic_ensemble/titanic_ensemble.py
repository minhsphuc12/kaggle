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

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
