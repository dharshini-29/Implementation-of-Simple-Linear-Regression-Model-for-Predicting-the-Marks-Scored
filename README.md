**Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored**

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict regression for marks by representing in a graph.
6.Compare graphs and hence linear regression is obtained for the given datas.

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train, regressor.predict(x_train),color='blue') 
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train, regressor.predict(x_train),color='red')
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE=',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE=',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: K Dharshini
RegisterNumber: 212223220017

```

## Output:
Head Values
![Screenshot 2025-02-28 094252](https://github.com/user-attachments/assets/5a17c401-f560-4193-89d8-c7b0523df028)
Tail Values
![image](https://github.com/user-attachments/assets/1b91e156-b944-446c-b165-0cbc1d4811f2)
Compare Dataset
![Screenshot 2025-02-28 094408](https://github.com/user-attachments/assets/e395b609-97be-45a6-8fec-71940b3eee83)
Predicted values
![Screenshot 2025-02-28 094526](https://github.com/user-attachments/assets/1a0a0f94-e3f8-478d-9663-1884f8601ca1)

Training Set
![Screenshot 2025-02-28 094613](https://github.com/user-attachments/assets/bbff8817-3377-4073-9905-6ce88a5474d1)
Testing Set
![Screenshot 2025-02-28 094642](https://github.com/user-attachments/assets/80c96868-2d4b-4e72-b1ed-448abee776ab)
Error
![Screenshot 2025-02-28 094723](https://github.com/user-attachments/assets/4c43d110-852e-4df1-b651-4571db457133)











## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
