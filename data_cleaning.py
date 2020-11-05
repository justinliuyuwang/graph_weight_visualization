# This is a script that cleans and calculates mean & stdev from raw distance measurements between items.

import glob
import pandas as pd
import scipy.io as sp
from statistics import mean, stdev
from pathlib import Path

#get filepaths for each of the categories' folders
categories = glob.glob('inputs/rawdata/*_raw')


outputfile = "inputs/cleandata/distance_data.csv"

#create file if it doesn't exist
if not Path(outputfile).is_file():
    f = open(outputfile, "x")
    f.close()

#flag marks whether it's the first time writing data to the output file
flag = 0

#for each category
for directory_path in categories:

    #category name extracted from folder path
    category = directory_path.replace("inputs/rawdata/", "").replace("_raw", "")

    #list of files in the category folder
    files = glob.glob(directory_path + '/*.mat')

    #dictionary to hold each item pair's list of distance measurements
    myDict = {}

    #for each file (participant response for this category)
    for file_path in files:

        #load matlab data
        mat_contents = sp.loadmat(file_path)
        data = mat_contents

        #clean \ufeff and whitespace from item names
        for index in range(len(data["stimuli"])):
            data["stimuli"][index] = data["stimuli"][index].replace("\ufeff", "").replace(" ", "")


        n = len(data["stimuli"])
        j = 0
        k = 1
        m = 0

        #iterate through all possible pairs of items and assign the appropriate measurements to each combination
        #j is the index of the first item and k is the index of the second item
        #m is the index of the measurement
        while j < n - 1:
            k = j + 1

            while k < n:
                #combine items to give item pair name
                item_pair = data['stimuli'][j] + '_AND_' + data['stimuli'][k]

                #if item pair hasn't been encountered yet then create an entry for the pair
                if item_pair not in myDict:
                    myDict[item_pair] = []

                #append the distance for this pair to the list of distances for this pair
                myDict[item_pair].append(data["rdmutv"][0][m])
                k += 1
                m += 1
            j += 1


    first_item_list = []
    second_item_list = []
    mean_distance_list = []
    SD_list = []
    category_list = []

    #for each pair of items, fill their data into the above lists
    for key in myDict:

        #if the item pair exists but in reverse order then we have an error in our data
        if key.split('_AND_')[1] + '_AND' + key.split('_AND_')[0] in myDict:
            print('ERROR WITH DATA - reversed pairs exist. raw data files likely contain items in different orders')
            exit()


        category_list.append(category)
        first_item_list.append(key.split('_AND_')[0])
        second_item_list.append(key.split('_AND_')[1])
        mean_distance_list.append(mean(myDict[key]))
        SD_list.append(stdev(myDict[key]))

    #fill in data frames
    data = {'Category': category_list,
            'Unique_pair_item1': first_item_list,
            'Unique_pair_item2': second_item_list,
            'Distance_m': mean_distance_list,
            'Distance_sd': SD_list
            }

    df = pd.DataFrame(data, columns=['Category', 'Unique_pair_item1', 'Unique_pair_item2', 'Distance_m', 'Distance_sd'])


    #export data frames to output file
    if flag == 1:
        df.to_csv("inputs/cleandata/distance_data.csv", index=False, mode='a', index_label=False, header=False)
    else:
        df.to_csv("inputs/cleandata/distance_data.csv", index=False, mode='a')
        flag = 1


