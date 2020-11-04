# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import glob
import pandas as pd
import scipy.io as sp
from statistics import mean, stdev
from pathlib import Path

categories = glob.glob('inputs/rawdata/*_raw')


my_filename = "inputs/cleandata/distance_data.csv"
if not Path(my_filename).is_file():
    f = open(my_filename, "x")
    f.close()

for directory_path in categories:

    category = directory_path.replace("inputs/rawdata/", "").replace("_raw", "")

    files = glob.glob(directory_path + '/*.mat')

    myDict = {}


    for file_path in files:

        mat_contents = sp.loadmat(file_path)
        data = mat_contents

        for index in range(len(data["stimuli"])):
            data["stimuli"][index] = data["stimuli"][index].replace("\ufeff", "").replace(" ", "")


        #print(data["stimuli"])
        #print(data["rdmutv"])

        #for index in range(len(data["rdmutv"][0])):
        #    print(data["rdmutv"][0][index])


        n = len(data["stimuli"])
        j = 0
        k = 1
        m = 0

        while j < n-1:
            k = j+1
            while k < n:
                item_pair = data['stimuli'][j] + '_AND_' + data['stimuli'][k]
                if item_pair not in myDict:
                    myDict[item_pair] = []

                myDict[item_pair].append(data["rdmutv"][0][m])
                k += 1
                m += 1
            j += 1

    #print(myDict)

    first_item_list = []
    second_item_list = []
    mean_distance_list = []
    SD_list = []
    category_list = []
    for key in myDict:

        if key.split('_AND_')[1] + '_AND' + key.split('_AND_')[0] in myDict:
            print('ERROR WITH DATA - reversed pairs exist. raw data files likely contain items in different orders')
            exit()
        category_list.append(category)
        first_item_list.append(key.split('_AND_')[0])
        second_item_list.append(key.split('_AND_')[1])
        mean_distance_list.append(mean(myDict[key]))
        SD_list.append(stdev(myDict[key]))


    data = { 'Category': category_list,
             'Unique_pair_item1': first_item_list,
             'Unique_pair_item2': second_item_list,
             'Distance_m': mean_distance_list,
             'Distance_sd': SD_list
    }

    df = pd.DataFrame(data, columns=['Category', 'Unique_pair_item1', 'Unique_pair_item2', 'Distance_m', 'Distance_sd'])

    df.to_csv("inputs/cleandata/distance_data.csv", index=False, mode='a')
