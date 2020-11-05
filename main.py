# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS

df = pd.read_csv('inputs/cleandata/distance_data.csv')
print(df)

category_list = df['Category'].unique()
print(category_list)

for category in category_list:
    print('\n\n\n')
    print('Category: ' + category + '\n')
    current_category_data = df.loc[df['Category'] == category, ['Unique_pair_item1', 'Unique_pair_item2', 'Distance_m']]

    unique_items_in_category = (current_category_data['Unique_pair_item1'].append(current_category_data['Unique_pair_item2'])).unique()
    print(current_category_data)

    print('Unique items: ')
    print(unique_items_in_category)

    unique_item_count = len(unique_items_in_category)

    distance_matrix = np.zeros(shape=(unique_item_count, unique_item_count))
   # print(distance_matrix)
    #print(type(distance_matrix))

    distance_matrix[np.triu_indices(distance_matrix.shape[0], 1)] = np.nan
    print(distance_matrix)

    matrix_df = pd.DataFrame(data=distance_matrix, index=unique_items_in_category, columns=unique_items_in_category)
    print(matrix_df)

    matrix_df_for_force_graph = matrix_df.copy()

    for index, row in current_category_data.iterrows():
        matrix_df.at[row['Unique_pair_item1'], row['Unique_pair_item2']] = row['Distance_m']
        matrix_df.at[row['Unique_pair_item2'], row['Unique_pair_item1']] = row['Distance_m']

        if row['Distance_m'] < 0.07:
            matrix_df_for_force_graph.at[row['Unique_pair_item1'], row['Unique_pair_item2']] = row['Distance_m']
            matrix_df_for_force_graph.at[row['Unique_pair_item2'], row['Unique_pair_item1']] = row['Distance_m']
        else:
            matrix_df_for_force_graph.at[row['Unique_pair_item1'], row['Unique_pair_item2']] = 0
            matrix_df_for_force_graph.at[row['Unique_pair_item2'], row['Unique_pair_item1']] = 0

    #print(matrix_df)

    #plt heatmap
    myindex = unique_items_in_category
    columns = unique_items_in_category
    plt.pcolor(matrix_df)#, cmap='seismic'
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

    #seaborn cluster + dendrogram
    g = sns.clustermap(matrix_df)

    #plt.title(category + " dendrogram and clustered heatmap")
    plt.savefig('./figures/' + category + '_dendrogram_and_clustered_heat_map.png')
    plt.show()



    #force-directed graph
    import networkx as nx

    A = matrix_df
    G = nx.from_pandas_adjacency(matrix_df_for_force_graph)

    pos = nx.kamada_kawai_layout(G, weight='weight')

    nx.draw_kamada_kawai(G, with_labels=True)

    plt.savefig('./figures/' + category + '_force_directed_graph.png')

    plt.show()
