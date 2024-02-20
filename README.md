# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VARNIKA.P
RegisterNumber: 212223240170
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
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
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
### Dataset:
![Screenshot 2024-02-20 113050](https://github.com/23008344/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742655/f049c551-52d0-4ed6-b9b0-50af2b25116e)

### Head Value:
![Screenshot 2024-02-20 113107](https://github.com/23008344/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742655/af89d980-0904-4398-8474-4bbcbae311af)

### Tail Value:
![Screenshot 2024-02-20 113114](https://github.com/23008344/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742655/77ea9c52-c9ac-4d7d-a150-e4904c02721d)

### X and Y values:
![Screenshot 2024-02-20 113140](https://github.com/23008344/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742655/d94c4dc4-9500-410a-846a-47436e827f70)

### Predication values of X and Y:
![Screenshot 2024-02-20 113212](https://github.com/23008344/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742655/f0a9f58d-b4c6-45fd-813e-89a3a20aead5)

### MSE,MAE and RMSE:
![Screenshot 2024-02-20 113242](https://github.com/23008344/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742655/5f0fba08-0da4-4f58-ba7d-0fb368e7d51c)

### Training Set:
![Screenshot 2024-02-20 113218](https://github.com/23008344/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742655/bd232a6f-e61c-4746-b87e-38b18d499b35)

### Testing Set:
![Screenshot 2024-02-20 113237](https://github.com/23008344/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742655/ca80c7c8-d507-424f-aa85-5b51d413211c)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
