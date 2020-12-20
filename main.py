import numpy as np
from csv import reader
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import scale
import matplotlib.pyplot as mpl

def train_test_split(data, train_size, index=0):
    x1pos = 14
    x2pos = 5
    ypos = 2
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    train_size *= len(data)
    for i in data:
        if index < train_size:
            train_x.append([float(i[x1pos]), float(i[x2pos])])
            train_y.append(float(i[ypos]))
        else:
            test_x.append([float(i[x1pos]), float(i[x2pos])])
            test_y.append(float(i[ypos]))
        index += 1

    return train_x, train_y, test_x, test_y


def read_csv(file):
    with open(file, newline='') as data:
        return list(reader(data))

def model_eval(y_data, x_data, model):
    y = []
    yhat = []
    for i in range(len(y_data)):
        y.append(y_data[i])
        yhat.append(float(model.predict([[x_data[i]]])))
    return mse(y, yhat)



def main():
    data = read_csv('house_dataset.csv')
    data.pop(0)
    train_x, train_y, test_x, test_y = train_test_split(data, 0.70)
    # mpl.plot(train_x, train_y, 'ro')
    x = np.array(train_x)
    #x = x.reshape((x.size, 1))
    y = np.array(train_y)
    scale(x)
    scale(y)
    sgd = SGDRegressor(shuffle=False).fit(x, y)
    print(sgd.score(x, y))
    #print(sgd.predict([[2570]]))
    #print(model_eval(train_y, train_x, sgd))
    print(sgd.predict([[1180, 1946]]))
    mpl.plot(x, y, 'ro')
    mpl.plot(x, sgd.predict(x))
    mpl.show()

main()
