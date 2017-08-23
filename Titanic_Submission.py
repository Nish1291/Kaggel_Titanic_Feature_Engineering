# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:42:48 2017

@author: Nishant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
from sklearn.svm import SVC
import re as re      #for title/string

train = pd.read_csv('train.csv', header = 0, dtype = {'Age' : np.float64})
test = pd.read_csv('test.csv', header = 0, dtype = {'Age' : np.float64})
#print(test.head())

full_data = [train,test]
print(train.info())

##Feature engineering

print(train[['Pclass','Survived']].groupby(['Pclass'], as_index = False).mean())
print(train[['Sex','Survived']].groupby(['Sex'], as_index = False).mean())
#With the number of siblings/spouse and the number of children/parents 
#we can create new feature called Family Size.

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print(train[['FamilySize','Survived']].groupby(['FamilySize'], as_index = False).mean())

#categorize people to check whether they are alone in this ship or not.

for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print(train[['IsAlone','Survived']].groupby(['IsAlone'], as_index = False).mean())

#the embarked feature has some missing value. and we try to fill those with the most occurred value ( 'S' )

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
print(train[['Embarked','Survived']].groupby(['Embarked'], as_index = False).mean())

#Fare also has some missing value and we will replace it with the median. 
#then we categorize it into 4 ranges.

for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
print(train[['CategoricalFare','Survived']].groupby(['CategoricalFare'], as_index = False).mean())

#In Age we have plenty of missing values. 
# generate random numbers between (mean - std) and (mean + std). 
#then we categorize age into 5 range.
#So generating random numbers

for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['CategoricalAge'] = pd.cut(train['Age'], 5)
print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())


#Now we are going to play wit names
#inside this feature we can find the title of people.

def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
    
print(pd.crosstab(train['Title'], train['Sex']))

#so we have titles. let's categorize it and check the title impact on survival rate.

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
print(train[['Title', 'Survived']].groupby(['Title'], as_index = False).mean())

#Now Last and finial play with data, cleaning the data

for dataset in full_data:
    #Mapping Sex
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
    
    #Mapping titles
    title_mapping = {'Mr': 1,'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    #mapping embarked
    dataset['Embarked']= dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    #mapping fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].astype(int)

#Feature Selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)

print (train.head(10))

train = train.values
test  = test.values

Classifier = SVC()
Classifier.fit(train[0::,1::], train[0::,0])
result = Classifier.predict(test)
print(result)