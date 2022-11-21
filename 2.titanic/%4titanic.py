#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:19:29 2022

@author: zhaoql21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load data
##### Load train and Test set

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
IDtest = test["PassengerId"]

# Outlier detection 

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
        
# Drop outliers
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
dir(sns)
        
## Join train and test datasets in order to obtain the same number of features during categorical conversion
train_len = len(train)
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)        
        
# Fill empty and NaNs values with NaN
dataset = dataset.fillna(np.nan)

# Check for Null values
dataset.isnull().sum()
        
def missing_percentage(df):
    '''这个函数输出缺失值个数以及所占百分比
    '''
    total=df.isnull().sum().sort_values(ascending=False)
    percentage=round(df.isnull().sum().sort_values(ascending=False)/100,2)
    return pd.concat([total,percentage],keys=['total','percentage'],axis=1)

missing_percentage(train)
missing_percentage(test)

def variable_survived(df,feature):
    '''这个函数计算不同变量幸存者的人数'''
    total=pd.DataFrame(df.loc[:,feature].value_counts(dropna=False))
    percentage=pd.DataFrame(round(df.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2))
    all_by=df.groupby('Survived')[feature].value_counts()
    return pd.concat([total,percentage],keys=['total','percentage'],axis=1), all_by

#feature analysis
mask=np.zeros_like(train.corr(),dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
sns.set_style('whitegrid')
plt.subplots(figsize=(15,12))
sns.heatmap(train[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Survived']].corr(),\
            annot=True,mask=mask,cmap='RdBu',linewidths=.9,linecolor='white',fmt='.2f')
plt.title("Correlation among feature")
        
 #SibSP 家庭关系
plt.subplots(figsize=(15,12)) 
sns.barplot(x='SibSp',y='Survived',data=train,palette='muted')
plt.ylabel("survival probability")
      
#Parch
plt.subplots(figsize=(15,12)) 
sns.barplot(x='Parch',y='Survived',data=train,palette='muted')
plt.ylabel("survival probability")


#Age
pal={1:'seagreen',2:'gray'}
g=sns.FacetGrid(train,col='Survived',row='Sex',margin_titles=True,hue='Survived')
g=g.map(sns.distplot,"Age")
plt.subplots_adjust(top=0.90)

plt.figure(figsize=(15,8),)
ax=sns.kdeplot(data=train.loc[train['Survived']==0,'Age'],color='gray',shade=True,label='not survived')
ax=sns.kdeplot(data=train.loc[train['Survived']==1,'Age'],color='Blue',shade=True,label='survived')
plt.xlabel("age")
plt.ylabel("survived")
plt.title("age distribution survived")

#Fare
dataset['Fare'].isnull().sum()
sns.displot(dataset['Fare'],color='m',kde=True)
#票价对数转换
dataset['Fare']=dataset['Fare'].map(lambda i: np.log(i) if i>0 else 0)
sns.displot(dataset['Fare'],color='b',kde=True)
#sex
sns.barplot(x="Sex",y="Survived",data=train)

g=sns.FacetGrid(data=dataset,row='Sex',col='Survived',margin_titles=True,palette=pal)
g.map(plt.hist,'Age')

train.groupby('Sex')['Survived'].mean()

#Pclass
train.groupby('Pclass')['Survived'].value_counts()
train.groupby('Pclass')['Survived'].mean()

plt.subplots(figsize=(15,10))
sns.barplot(data=train,x='Pclass',y='Survived',capsize=.05,errcolor='blue',errwidth=3)
plt.title("Pclass of survived")
plt.xticks([0,1,2],['upper','middle','lower'])

#explore pclass an sex
plt.subplots(figsize=(15,10),dpi=200)
sns.barplot(data=train,x="Pclass",y='Survived',hue='Sex',palette='muted',capsize=.05,errcolor='blue',errwidth=3)
plt.title("Pclass of survived")
plt.xticks([0,1,2],['upper','middle','lower'])

#embarked
variable_survived(dataset, 'Embarked')
dataset[dataset['Embarked'].isnull()]
sns.set_style('darkgrid')
fig,ax=plt.subplots(figsize=(16,12),ncols=2,dpi=200)
ax1=sns.boxplot(data=train,x='Embarked',y='Fare',hue='Pclass',ax=ax[0])
ax2=sns.boxplot(data=test,x='Embarked',y='Fare',hue='Pclass',ax=ax[1])
ax1.set_title("train set")
ax2.set_title('test set')

#依据费用，将Embarked缺失值转换为"c"
train['Embarked'].fillna('C',inplace=True)
variable_survived(train, 'Embarked')

g=sns.FacetGrid(data=train,col='Embarked',palette='muted')
g.map(plt.hist, 'Pclass')

#缺失数据的填充
 #age
dataset['Age'].isnull().sum()
plt.figure(figsize=(10,8),dpi=250)
sns.boxplot(data=dataset,x="Sex",y='Age')

plt.figure(figsize=(10,8),dpi=300)
sns.boxplot(data=dataset,x="Sex",y='Age',hue='Pclass')

plt.figure(figsize=(10,8),dpi=250)
sns.boxplot(data=dataset,x="Parch",y='Age')

plt.figure(figsize=(10,8),dpi=250)
sns.boxplot(data=dataset,x="SibSp",y='Age')

#将性别转换为亚变量
dataset['Sex']=dataset['Sex'].map({'male':0,'female':1})

mask=np.zeros_like(dataset[['Age','Sex','SibSp','Parch','Pclass']].corr(),dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
plt.figure(figsize=(10,10),dpi=250)
sns.heatmap(dataset[['Age','Sex','SibSp','Parch','Pclass']].corr(), \
            cmap='BrBG',annot=True,fmt='.2g')


#填充年龄缺失变量
index_Nan_age=list(dataset['Age'][dataset['Age'].isnull()].index)

for i in index_Nan_age:
    age_med=dataset['Age'].median()
    age_pred=dataset['Age'][((dataset['SibSp']==dataset.iloc[i]['SibSp'])&\
                             (dataset['Parch']==dataset.iloc[i]['Parch'])&\
                             (dataset['Pclass']==dataset.iloc[i]['Pclass']))].median() 
    if not np.isnan(age_pred):
        dataset['Age'].iloc[i]=age_pred
    else:
        dataset['Age'].iloc[i]=age_med
        
variable_survived(dataset,'Age')  

plt.figure(figsize=(15,10),dpi=300)
sns.boxplot(data=dataset,x="Survived",y="Age",palette='muted')
plt.title("survived of age")

plt.figure(figsize=(15,10),dpi=300)
sns.violinplot(data=dataset,x="Survived",y="Age",palette='muted')
plt.title("survived of age")

'''特征工程'''
#name/title
dataset['Name'].sample(5)
#get title from name
dataset_title=[i.split(',')[1].split('.')[0].strip() for i in dataset['Name']]
dataset['Title']=pd.Series(dataset_title)
dataset['Title'].sample(10)

plt.figure(figsize=(15,20),dpi=300)
sns.countplot(x='Title',data=dataset)

dataset['Title'].value_counts()

#convert to categorical values of title
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)

dataset['Title'].value_counts().sort_values()

plt.figure(figsize=(10,8),dpi=200)
sns.countplot(x='Title',data=dataset)

plt.figure(figsize=(10,8),dpi=250)
sns.barplot(x='Title',y='Survived',data=dataset,capsize=.05,errcolor='blue',errwidth=3)
plt.title("The title of survived")
plt.xticks([0,1,2,3],['master','miss-mrs','Mr','rare']) 


#删除Name列
dataset.drop(labels='Name',axis=1,inplace=True)

#家庭人数
dataset['Fsize']=dataset['SibSp']+dataset['Parch']+1
plt.figure(figsize=(10,8),dpi=200)
sns.lineplot(data=dataset,x='Fsize',y='Survived')
plt.title("family size of survived")


#家庭人数亚变量生成
def family_group(Fsize):
    '''这个函数计算家庭分组'''
    a=''
    if (Fsize<=1) :
        a='Single'
    elif (Fsize<=2):
        a='SmallF'
    elif (Fsize<=4):
        a='MedF'
    else:
        a='LargeF'
    return a
dataset['family_group']=dataset['Fsize'].map(family_group)        

dataset['family_group'].value_counts()
plt.figure(figsize=(12,10),dpi=200)
sns.barplot(data=dataset,x='family_group',y='Survived', capsize=0.05)
plt.title("the survive of family size")
#one-hot
dataset=pd.get_dummies(data=dataset,columns=['family_group'],drop_first=False)
dataset.drop('Fsize',axis=1,inplace=True)


#title
dataset=pd.get_dummies(data=dataset,columns=['Title'],drop_first=False)
#embarked
dataset=pd.get_dummies(data=dataset,columns=['Embarked'],drop_first=False,prefix='Em')

#cabin
dataset['Cabin'].value_counts(dropna=False)
dataset['Cabin']=pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
dataset=pd.get_dummies(data=dataset,columns=['Cabin'],prefix='Cabin',drop_first=False)

#ticket
dataset['Ticket'].head()

Ticket=[]
for i in list(dataset['Ticket']):
    if not i.isdigit():
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])
    else:
        Ticket.append("X")
        
dataset['Ticket']=Ticket
dataset['Ticket'].head()

dataset=pd.get_dummies(data=dataset,columns=['Ticket'],prefix='T')

#Pclass
dataset['Pclass'].value_counts(dropna=False)
dataset['Pclass'].astype('category')
dataset=pd.get_dummies(data=dataset,columns=['Pclass'],prefix='Pc')
#drop useles variables
dataset.drop(labels=['PassengerId'],axis=1,inplace=True)
#view dataset
dataset.head()

#separate train dataset and test dataset
train=dataset[:train_len]
test=dataset[train_len:]
test.drop(labels=['Survived'],axis=1,inplace=True)
 
#separate train feature and label
train['Survived']=train['Survived'].astype(int)
Y_train=train['Survived']
x_train=train.drop(labels=['Survived'],axis=1)


#交叉验证模型
from sklearn.model_selection import StratifiedKFold
kfold=StratifiedKFold(n_splits=10)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    clf=make_pipeline(StandardScaler(),classifier)
    cv_results.append(cross_val_score(clf, x_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

plt.figure(figsize=(10,10),dpi=250)
sns.barplot(data=cv_res,y='Algorithm',x='CrossValMeans',palette="Set3",orient="h")

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")




### META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING

# Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(x_train,Y_train)

ada_best = gsadaDTC.best_estimator_

gsadaDTC.best_score_
  
