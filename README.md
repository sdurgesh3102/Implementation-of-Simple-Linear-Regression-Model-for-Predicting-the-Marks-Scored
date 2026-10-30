# Implementation of Simple Linear Regression Model for Predicting the Marks Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries (numpy, pandas, matplotlib, scikit-learn).
2. Load the dataset containing the hours studied (X) and marks scored (Y).
3. Split the dataset into training and testing sets.
4. Create a Linear Regression model and train it using the training data.
5. Predict the marks for the test data using the trained model.
6. Evaluate the model using MSE, MAE and RMSE.
7. Plot the training and testing sets with the regression line.

## Program:
```python
'''
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: <S.DURGESH>
RegisterNumber: <212225230064>
'''
(paste the code from below)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = {
    "Hours": [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5,
              3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 2.7, 4.8, 3.8, 6.9, 7.8],
    "Scores": [21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41,
               42, 17, 95, 30, 24, 67, 69, 30, 54, 35, 76, 86],
}
df = pd.DataFrame(data)

print(df.head())
print(df.tail())

X = df[["Hours"]].values
Y = df["Scores"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

reg = LinearRegression()
reg.fit(X_train, Y_train)

Y_pred = reg.predict(X_test)
print("Predicted:", Y_pred)
print("Actual   :", Y_test)

mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("MSE  =", mse)
print("MAE  =", mae)
print("RMSE =", rmse)
print("Slope =", reg.coef_[0], " Intercept =", reg.intercept_)

plt.scatter(X_train, Y_train, color="green")
plt.plot(X_train, reg.predict(X_train), color="red")
plt.title("Training set (Hours vs Scores)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, reg.predict(X_test), color="red")
plt.title("Test set (Hours vs Scores)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
<img width="626" height="571" alt="image" src="https://github.com/user-attachments/assets/8ee8092c-5f33-450e-92bd-4b5ce1a636e6" />
<img width="541" height="332" alt="image" src="https://github.com/user-attachments/assets/83b387b1-e30b-4fdb-8399-adb2344faeca" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
