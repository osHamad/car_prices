import numpy as np
import csv
import setup
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import minmax_scale


def train_test(x, y, train_size=0.70, index=0):
    train_x = x[:round(train_size * len(y))]
    test_x = x[round(train_size * len(y)):]
    train_y = y[:round(train_size * len(y))]
    test_y = y[round(train_size * len(y)):]

    return train_x, train_y, test_x, test_y

def main():
    # setting up our data and removing unwanted instances
    with open('auto_mobile_data.csv') as file:
        data = list(csv.reader(file))
        data = setup.remove_inst(data)

    # creating a set of attributes to skip
    skip_atr = setup.skip_attribute(data)

    # adding attributes with no correlation to skip_atr (see scatter_plots\analysis.txt)
    # remove: symboling, losses, car height, bore, stroke, compression ratio, peak rpm
    skip_atr.update({0, 1, 12, 18, 19, 20, 22, 25})

    # dictionary of the mean of values of attributes
    mean_nums = setup.missing_values(data)

    # arranging the x and y data
    # x is a 2d list and y is 1d
    x = []
    y = [float(i[len(i)-1]) for i in data]
    for i in data:
        thing = [i[val] if i[val] != '?' else mean_nums[val] for val in range(len(i))]
        x.append([float(thing[val]) for val in range(len(thing)) if val not in skip_atr])

    # splitting our data into training and testing data
    train_x, train_y, test_x, test_y = train_test(x, y)

    # preparing data for regression
    x = np.array(train_x)
    y = np.array(train_y)
    x = minmax_scale(x)
    y = minmax_scale(y)

    # fitting data to our model
    sgd = SGDRegressor().fit(x, y)

    # scoring our model
    # we must score our model with unseen data to prevent over/under fitting
    test_x = minmax_scale(test_x)
    test_y = minmax_scale(test_y)
    print('score of model: ', sgd.score(test_x, test_y))

main()
