import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('https://raw.githubusercontent.com/YBIFoundation/Dataset/main/MPG.csv')
df.head()
df.nunique
df.info()
df.describe()
df.corr()
df=df.dropna()
df.info()
sns.pairplot(df,x_vars=['displacement','horsepower','weight','acceleration','mpg'],y_vars=['mpg'])
sns.regplot(x='displacement',y='mpg',data=df)
df.columns
y=df['mpg']
y.shape
x=df[['displacement','horsepower','weight','acceleration']]
x.shape
x
y
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x=ss.fit_transform(x)
x
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=42)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
lr.intercept_
lr.coef_
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score
mean_absolute_error(y_test,y_pred)
mean_absolute_percentage_error(y_test,y_pred)
r2_score(y_test,y_pred)
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
x_train2=poly.fit_transform(x_train)
x_test2=poly.fit_transform(x_test)
lr.intercept_
lr.coef_
y_pred1=lr.predict(x_test2)
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score
mean_absolute_error(y_test,y_pred1)
mean_absolute_percentage_error(y_test,y_pred1)
r2_score(y_test,y_pred1)
