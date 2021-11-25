import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import random

def main():
  array = np.loadtxt('case_study2_data/data.txt') #loading data
  list1 = array.tolist()
    
  x_train = []
  y_train = []

  x_test = []
  y_test = []
  
  #splitting into training and testing datasets
  for row in list1:
      label = row.pop()
      if random.random() <= 0.5:
        x_train.append(row)
        y_train.append(label)
      else:
        x_test.append(row)
        y_test.append(label)
    
  x_train = np.array(x_train)
  y_train = np.array(y_train)
  x_test = np.array(x_test)
  y_test = np.array(y_test)
  
  #creating model
  regr = linear_model.LinearRegression(normalize=True)

  regr.fit(x_train, y_train)

  pred = regr.predict(x_test)

  # coefficients
  print('Coefficients: \n', regr.coef_)
  #error
  print('Mean squared error: %.2f'
    % mean_squared_error(y_test, pred))
  # The coefficient of determination: 1 is perfect prediction
  print('Coefficient of determination: %.2f'
    % r2_score(y_test, pred))
  
main()