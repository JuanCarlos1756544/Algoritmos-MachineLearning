import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score


def reg_iter(i):
  #FuelConsumption
  df = pd.read_csv("precio.csv")

  # # Info del dataframe
  # df.head()
  # df.describe()

  # Dividir los datos para entrenar y evaluar 80 - 20
  msk = np.random.rand(len(df)) < 0.5
  train = df[msk]
  test = df[~msk]

  # Entrenar el modelo 
  modelo = linear_model.LinearRegression()
  train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
  train_y = np.asanyarray(train[['CO2EMISSIONS']])
  modelo.fit (train_x, train_y)
  t0 = modelo.intercept_[0]
  t1 = modelo.coef_[0][0]

  # # Mostrar el modeo
  # print ('theta1: ', t1)
  # print ('theta0: ', t0)

  # Evaluar el modelo
  test_x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
  test_y = np.asanyarray(test[['CO2EMISSIONS']])
  test_y_hat = modelo.predict(test_x)

  mae=np.mean(np.absolute(test_y_hat - test_y))
  mse=np.mean((test_y_hat - test_y) ** 2)
  r2= r2_score(test_y_hat , test_y)
  # print("El modelo y={} + {}x ".format(t0,t1))
  # print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
  # print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
  # print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )


  plt.scatter(train_x, train_y,  color='#e0d6ff')
  plt.scatter(test_x, test_y,  color='#bcbdf6')
  plt.plot(train_x, modelo.coef_[0][0]*train_x + modelo.intercept_[0], color='#ffd9fa')
  plt.xlabel('ENGINESIZE')
  plt.ylabel('CO2EMISSIONS')
  plt.clf()

  return t0, t1, mae,mse,r2



def main():
  print("i,t0,t1,mae,mse,r2")
  print(reg_iter(1))


if __name__ == "__main__":
  main()
