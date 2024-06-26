# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the standard liabraries.

2.upload the dataset and check for any null or duplicated values using.isnull() and .duplicated() function respectively

3.import LabelEncoder and encode the dataset

4.Import LogisticRegression from sklearn and aplly the model on the dataset

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by imporving the required modules from sklearn.

7.Apply new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: yogaraj S
RegisterNumber:  212223040248
*/
```
```
import pandas as pd
data = pd.read_csv("/content/Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("accuracy score:\n",accuracy)
print("confusion_matrix:\n",confusion)
print("classification_report:\n",cr)
```
## Output:

![Screenshot 2024-03-12 162033](https://github.com/yogaraj2/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/153482637/e254d567-3441-43a0-9ba0-4b124150350a)

![Screenshot 2024-03-12 162053](https://github.com/yogaraj2/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/153482637/6d882639-2a65-49c9-8909-bcf8013c0563)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
