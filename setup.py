import statistics as stat

# we will setup the data to ensure we get optimal results
# for our regression algorithm, we must skip over all attributes with string values
# we should remove any instances with no Price attribute, since it is our y var
# we should check to see if any other instances are empty
# if that is the case, we should fill them in with the average of the other instances of the same attribute


# skipping over attributes with discrete values
def skip_attribute(data):
    remove_atr = set()
    for i in data:
        for j in range(len(i)):
            if i[j].isalpha():
                remove_atr.add(j)
    return remove_atr


# removing instances with no price attribute
def remove_inst(data):
    removed_inst = [i for i in data if i[len(i)-1] != '?']
    return removed_inst


# giving values to missing attributes of instances
def missing_values(data):
    mean_num = {}
    for i in range(len(data[0])):
        try:
            mean_num[i]=stat.mean([float(x[i]) for x in data if x[i] != '?'])
        except ValueError:
            mean_num[i]='string'
    return mean_num
