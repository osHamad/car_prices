import statistics as stat

# we will setup the data to ensure we get optimal results
# for regression algorithms, it is better to use continuous data so we will take out discrete attributes
# we should remove any instances with no Price attribute, since it is our y var
# we should check to see if any other instances are empty
# if that is the case, we should fill them in with the average of the other instances of the same attribute

def format_data(data):
    # taking out discrete values
    remove_atr = set()
    for i in data:
        for j in range(len(i)):
            if i[j].isalpha():
                remove_atr.add(j)
    data = [[y for y in x if x.index(y) not in remove_atr] for x in data]

    # removing instances with no price attribute
    for i in data:
        if i[len(i)-1] == '?':
            data.remove(i)

    # giving values to missing attributes of instances
    mean_num = {}
    for i in range(len(data[0])):
        mean_num[i] = stat.mean([float(x[i]) for x in data if x[i] != '?'])

    for i in data:
        for j in range(len(i)):
            if i[j] == '?':
                i[j] = mean_num[j]

    return data



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
    for i in data:
        if i[len(i)-1] == '?':
            data.remove(i)
    return data


# giving values to missing attributes of instances
def missing_values(data):
    mean_num = {}
    for i in range(len(data[0])):
        try:
            mean_num[i] = stat.mean([float(x) for x in data[i] if x != '?'])
        except ValueError:
            pass
    return mean_num
