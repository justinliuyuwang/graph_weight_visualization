# This is a script that visualizes the distances between items of each category


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import networkx as nx

#import data
df = pd.read_csv('inputs/cleandata/distance_data.csv')

#list of categories
category_list = df['Category'].unique()

#create visualizations for each category
for category in category_list:

    #get distance info for this category
    current_category_data = df.loc[df['Category'] == category, ['Unique_pair_item1', 'Unique_pair_item2', 'Distance_m']]

    #get unique items
    unique_items_in_category = (current_category_data['Unique_pair_item1'].append(current_category_data['Unique_pair_item2'])).unique()
    unique_item_count = len(unique_items_in_category)

    #zero distance matrix of the appropriate size
    distance_matrix = np.zeros(shape=(unique_item_count, unique_item_count))

    #replace half triangle of the matrix with NaNs
    distance_matrix[np.triu_indices(distance_matrix.shape[0], 1)] = np.nan

    #make a dataframe using the above numpy matrix
    matrix_df = pd.DataFrame(data=distance_matrix, index=unique_items_in_category, columns=unique_items_in_category)

    #create another dataframe for the force-directed graph
    matrix_df_for_force_graph = matrix_df.copy()

    #fill the dataframes with the measurement data for this category
    for index, row in current_category_data.iterrows():

        #most of the visualizations can use a symmetric distance matrix (dataframe)
        matrix_df.at[row['Unique_pair_item1'], row['Unique_pair_item2']] = row['Distance_m']
        matrix_df.at[row['Unique_pair_item2'], row['Unique_pair_item1']] = row['Distance_m']

        #the force-directed graph requires a threshold. relationships whose distance falls under the threshold will be drawn
        #without a threshold, the graph becomes a web connecting all items
        if row['Distance_m'] < 0.07:
            matrix_df_for_force_graph.at[row['Unique_pair_item1'], row['Unique_pair_item2']] = row['Distance_m']
            matrix_df_for_force_graph.at[row['Unique_pair_item2'], row['Unique_pair_item1']] = row['Distance_m']
        else:
            #if the distance between two items is too large, then set their connecting edge, or relationship to 0 (non-existent)
            matrix_df_for_force_graph.at[row['Unique_pair_item1'], row['Unique_pair_item2']] = 0
            matrix_df_for_force_graph.at[row['Unique_pair_item2'], row['Unique_pair_item1']] = 0


    #heatmap (matplotlib)
    myindex = unique_items_in_category
    columns = unique_items_in_category
    plt.pcolor(matrix_df)
    plt.yticks(np.arange(0.5, len(myindex), 1), myindex)
    plt.xticks(np.arange(0.5, len(columns), 1), columns, rotation='vertical')
    plt.colorbar()
    ax = plt.gca()
    ax.invert_yaxis()
    plt.title(category + " heatmap")
    plt.subplots_adjust(left = 0.262, bottom = 0.26, right = 0.857)
    plt.savefig('./figures/'+category + '_heatmap.png')
    plt.show()


    #dendrogram
    dists = squareform(matrix_df)
    linkage_matrix = linkage(dists, "single")
    dendrogram(linkage_matrix, labels=unique_items_in_category)
    plt.title(category + " dendrogram")
    plt.xticks(rotation='vertical')
    plt.subplots_adjust(left=0.262, bottom=0.26, right=0.857)
    plt.savefig('./figures/'+category + '_dendrogram.png')
    plt.show()


    #cluster + dendrogram (seaborn)
    g = sns.clustermap(matrix_df)
    plt.savefig('./figures/' + category + '_dendrogram_and_clustered_heat_map.png')
    plt.show()


    #force-directed graph
    A = matrix_df
    G = nx.from_pandas_adjacency(matrix_df_for_force_graph)
    pos = nx.kamada_kawai_layout(G, weight='weight')
    nx.draw_kamada_kawai(G, with_labels=True)
    plt.savefig('./figures/' + category + '_force_directed_graph.png')
    plt.show()