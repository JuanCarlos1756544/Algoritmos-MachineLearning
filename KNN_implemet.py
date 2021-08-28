
#Alumno:Juan Carlos Arredondo Herrera 

import pandas as pd
import math
import statistics
from statistics import mode

#funcion especifica para calcular la distancia de los 4 atributos
def eucledian_distance(x, y, z, j):
    dx = x[0] - x[1]
    dy = y[0] - y[1]
    dz = z[0] - z[1]
    dj = j[0] - j[1]

    return math.sqrt(dx**2 + dy**2 + dz**2 + dj**2)

data_file = 'iris.csv'
df = pd.read_csv(data_file)
number_of_training_points = math.floor(0.8*len(df))

train_data = df[:number_of_training_points]
test_data = df[number_of_training_points:]

# definicion de k
k = 4

#proceso para predecir la clase dependiendo l el valor de k
correct_count = 0
for test_index, test_row in test_data.iterrows():
    x1 = test_row["sepal_length"]
    y1 = test_row["petal_length"]
    z1 = test_row["sepal_width"]
    j1 = test_row["petal_width"]

    neighbors = {}

    for index, row in train_data.iterrows():
        x2 = row["sepal_length"]
        y2 = row["petal_length"]
        z2 = row["sepal_width"]
        j2 = row['petal_width']

        x = (x1, x2)
        y = (y1, y2)
        z = (z1, z2)
        j = (j1, j2)

        # computing the eucledian distance
        distance = eucledian_distance(x, y, z, j)
        neighbors[distance] = row["species"]
    top_k = []

    count = 0

    for key in sorted(neighbors.keys()):
        if len(top_k) != k:
            top_k.append(neighbors[key])
    # print(top_k)
    prediction = ''
    try:
        prediction = mode(top_k)
    except statistics.StatisticsError:
        prediction = top_k[0]

    if (prediction == test_row["species"]):
        correct_count += 1

    print(test_row["species"])
    print(prediction)
    print(correct_count)

print('Accuracy of model: ' + "{0:.1%}".format(correct_count/len(test_data)))