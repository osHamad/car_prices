import csv
import setup
import matplotlib.pyplot as plt

# in this file, we will test our data
# we will graph our data and pick the attributes that connect with y
# our y (dependant var) will be the car prices

# data attributes:
atr = {0:'symboling', 1:'losses', 2:'brand', 3:'fuel type', 4:'asparation', 5:'doors',
6:'body style', 7:'wheel drive', 8:'engine location', 9:'wheel base', 10:'car length',
11:'car width', 12:'car height', 13:'car weight', 14:'engine', 15:'cylinders',
16:'engine size', 17:'fuel system', 18:'bore', 19:'stroke', 20:'compression ratio',
21:'horsepower', 22:'peak rpm', 23:'city miles per gallon', 24:'highway miles per gallon', 25:'price'}

def main():
    # setting up our data
    with open('auto_mobile_data.csv') as file:
        data = list(csv.reader(file))
        print(len(data))
        data = [i[len(i)-1] for i in data if i[len(i)-1] != '?']
        print(len(data))

    skip_atr = setup.skip_attribute(data)

    mean_nums = setup.missing_values(data)

    # plotting each attribute in terms of y
    for index in range(len(data[0])):
        if index not in skip_atr:
            x = [float(i[index]) if i[index] != '?' else float(mean_nums[index]) for i in data]
            y = [float(i[len(i)-1]) for i in data]
            print(x, y)
            plt.plot(x, y, 'bo')
            plt.xlabel(atr[index])
            plt.ylabel(atr[25])
            plt.axis([min(x), max(x), min(y), max(y)])
            plt.yscale('linear')
            plt.xscale('linear')
            plt.title(f'{atr[25]} in terms of {atr[index]}')
            plt.show()

if __name__ == '__main__':
    main()
