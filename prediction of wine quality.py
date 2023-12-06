#importinng libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings(action='ignore')

#Loading Dataset:
df=pd.read_csv('Winequality-red.csv')
print(df.head())

#Description:
print(df.describe(include='all'))

#Finding Null Values:
print(df.isnull().sum())
print(df.corr())

#quality mean values:
print(df.groupby('quality').mean())

#Data Analysis:
sns.countplot(x='quality',data=df)
plt.show()

sns.countplot(x='pH',data=df)
plt.show()

sns.countplot(x='alcohol',data=df)
plt.show()

sns.countplot(x='fixed acidity',data=df)
plt.show()

sns.countplot(x='volatile acidity',data=df)
plt.show()

sns.countplot(x='citric acid',data=df)
plt.show()

sns.countplot(x='density',data=df)
plt.show()

#dist plot:
sns.distplot(df['alcohol'])
plt.show()

g=df.plot(kind='box',subplots=True,layout=(4,4),sharex=False)
print(g)
plt.show()
f=df.plot(kind='density',subplots=True,layout=(4,4),sharex=False)
print(f)
plt.show()

#histogram:
df.hist(bins=50)
plt.show()

#heatmap for expressing correlation:
corr=df.corr()
sns.heatmap(corr,annot=True)
plt.show()

#pair plot: dont use this
'''sns.pairplot(df)
plt.show()'''

#feature selection:
df['good quality']=[1 if x>=7 else 0 for x in df['quality']]
#Separate feature variables and target variable
x=df.drop(['quality','good quality'], axis=1)
y=df['good quality']
#see proportion of good vs bas wines
pro=df['good quality'].value_counts()
print(pro,y)

#splitting dataset
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,)

#Logistic Regression:
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
from sklearn.metrics import accuracy_score
print('Accuracy Score Linear Regression: ',accuracy_score(ytest,ypred))

#using KNN:
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain,ytrain)
y1pred=model.predict(xtest)
print('Accuracy score KNN: ',accuracy_score(ytest,y1pred))

#using SVM
from sklearn.svm import SVC
model=SVC()
model.fit(xtrain,ytrain)
y2pred=model.predict(xtest)
print('Accuracy score SVM: ',accuracy_score(ytest,y2pred))

#using GaussianNB:
from sklearn.naive_bayes import GaussianNB
model1=GaussianNB()
model1.fit(xtrain,ytrain)
y3pred=model1.predict(xtest)
print('Accuracy score GaussianNB: ',accuracy_score(ytest,y3pred))

#comaprison of accuracy
result=pd.DataFrame({'Model':['Logistic Regression','KNN','SVM','GaussianNB'],
                     'Score':[0.860416,0.85,0.8666,0.82083]})
final=result.sort_values(by='Score',ascending=False)
print(final)




