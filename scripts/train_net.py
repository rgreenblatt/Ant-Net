import numpy as np
import re
from sklearn.model_selection import train_test_split
import pandas as pd

def get_label_from_ID(length, ID):
    parsed = ID.replace('ant_data__', '')

    states = np.empty((length), dtype=float)

    for i in range(length):
        vec_string = re.sub(r'__.*', '', parsed)

        first = re.sub(r'_.*', '', vec_string)
        #states[i][0] = int(first)

        first += '_'
        second = re.sub(first, '', vec_string)
        states[i] = int(second)

        if(i != 6 - 1):
            remove_vec = vec_string + "__"
            parsed = re.sub(remove_vec, '', parsed)

    return states


length = 6 #This is used as the labels input as that gets provides to the get_label_from_ID func

print("Reading in names list...")
id_list = pd.read_csv("data_npy/names.txt").values
dim = (25, 25)
n_channels = 1
y_dtype=float

id_list_train, id_list_test = train_test_split(id_list, test_size=0.20)

train_id_dict = {}
test_id_dict = {}

print("Generating dictionary of train labels...")
for id_train in id_list_train:
    train_id_dict[id_train[0]] = get_label_from_ID(length, id_train[0])

print("Generating dictionary of test labels...")
for id_test in id_list_test:
    test_id_dict[id_test[0]] = get_label_from_ID(length, id_test[0])

print("Beginning actual training")

#Long term ideas:
#We want the last layer to be a Dense output with the same number as there are states. Activation elu.
#we will treat each of the different states as a class and use the net as a "classifier"

#Short term ideas:
#classify forward allowed vs not allowed
