# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function. 

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Adchayakiruthika M S
RegisterNumber: 212223230005

import pandas as pd
data=pd.read_csv("/content/Employee_EX6.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level", "last_evaluation", "number_project","average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## Head:
![image](https://github.com/Adchayakiruthika18/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139995/9bcdd103-6ca6-4fb0-bd8d-35046ff98af9)

## Data.info():
![image](https://github.com/Adchayakiruthika18/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139995/a7cecd0e-ac9e-4ba1-8535-cb6bfae2cd15)

## isnull() and sum():
![image](https://github.com/Adchayakiruthika18/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139995/5ad25940-b587-412e-bd22-e0c0a8e51322)

## Data Value Counts:
![image](https://github.com/Adchayakiruthika18/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139995/066d921d-6ed8-496f-a839-ce8cd5c08f06)

## Data.head() for salary:
![image](https://github.com/Adchayakiruthika18/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139995/8a7c939c-c314-42bd-8b34-fdc0232d2ff2)

## X.head:
![image](https://github.com/Adchayakiruthika18/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139995/9f9408c5-b954-400c-bd4a-09ab9b293758)

## Accuracy Value:
![image](https://github.com/Adchayakiruthika18/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139995/4887326c-7e40-4f7b-9cd2-a403a708cbb2)

## Data Prediction:
![image](https://github.com/Adchayakiruthika18/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139995/c4b39c95-ae8f-4067-a907-7056b5db1f66)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
