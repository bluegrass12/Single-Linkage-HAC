#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
___author___ = "Carl Youel"
# version ='1.0'
# ---------------------------------------------------------------------------


"""
SingleLinkageHAC: Implements Single Linkage Algorithm for Hierarchical 
Agglomerative Clustering of a dataset

This module has a similar functionality to Scipy's Linkage module, clustering 
data using the Single Linkage algorithm: each step, combine closest elements 
into a single cluster.  Tiebreaking is done by choosing the pair of vectors 
with smaller indices in the distance matrix (i.e. the pair found first). 
This is from where variations from Scipy's Linkage module will arise.  

Input: 
- CSV file 
- # of CSV rows to include (enter "-1" to include all rows)
- features to include (enter "all" to include all features)

Output: 
- Z.npy, a numpy array representing the hierarchical clustering encoded as a 
    linkage matrix
- a dendrogram visualizing the hierarchical clustering

Usage: python3 SingleLinkageHAC <csvfile> <# of CSV rows to include> 
        <feature1> <...> <feature x> 

Notes: 
- Features must be numeric. Nonnumeric features will be removed and won't 
    impact clustering
- This module is slower and less efficient than Scipy's Linkage module

"""


# ---------------------------------------------------------------------------
import csv
import sys
import os
import numpy as np
from scipy.cluster.hierarchy import dendrogram as DG
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------


def load_data(filepath):
    """
    Creates list of dictionaries, where each dictionary represents a point.

    Parameters:
    filepath (string): name of CSV file to load from

    Returns:
    list: list of dictionaries

    """
    with open(filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
    return data

def filter_features(element, features):
    """
    Filters data to contain features listed as arguments.
    Features must be a numeric! If not numeric, will be removed from features 

    Parameters:
    element (dict): a single datapoint vector
    features (list): features to be included in feature vector

    Returns:
    numpy array: a (6,) array containing filtered vector values

    """
    filtered_dict = {}
    for feature in features:
        filtered_dict[feature] = element.get(feature)

    k = list(filtered_dict.values())

    # if any features given do not exist, raise exception
    for item in k:
        if item is None:
            raise ValueError

    # item must be numeric
    to_remove = []
    for item in k:
        if not item.isnumeric():
            to_remove.append(item)

    for item in to_remove:
        k.remove(item)
    if len(k) == 0:
        print('Provided features are nonnumeric. Only numeric feature are accepted.')
        sys.exit(1)

    filtered_array = np.array(k, dtype='int64')
    return filtered_array

def single_linkage(features):
    """
    Calculates hierarchical clustering of dataset using Single Linkage 
    algorithm.

    Using cluster_forest to represent current clusters and distance_matrix to 
    represent the distances between all vectors, iteratively forms additional 
    clusters using Single Linkage algorithm.

    Parameters:
    features (list): list of numpy arrays, with each array representing a 
    single feature vector

    Returns:
    numpy array: the linkage matrix representing Single Linkage hierarchical 
    clustering of dataset

    """
    cluster_forest = []
    # Create cluster forest (list of lists)
    for i in range(len(features)):
        cluster_forest.append([i])

    # Create dimension matrix (n x n)
    distance_matrix = calc_dist_mtrx(features)

    # Create cluster hierarchy matrix
    hierarchy = np.zeros([len(features) - 1, 4])
    for row in hierarchy:
        minimum = find_min_overall(cluster_forest, distance_matrix)
        # First element = smallest element
        minval = min(minimum.get('P1'), minimum.get('P2'))
        maxval = max(minimum.get('P1'), minimum.get('P2'))
        row[0] = minval
        row[1] = maxval
        row[2] = minimum.get('Distance')
        row[3] = minimum.get('Number')

        # Modify cluster forest
        new_cluster = cluster_forest[int(row[0])] + cluster_forest[int(row[1])]
        cluster_forest[int(row[1])] = [-1]
        cluster_forest[int(row[0])] = [-1]
        # -1: indicator that cluster has been absorbed
        cluster_forest.append(new_cluster)

    return hierarchy

def calc_dist_mtrx(features):
    """
    Calculate distance matrix, containing distances between all vectors

    Parameters:
    features (list): list of numpy arrays, with each array representing a 
    single feature vector

    Returns:
    numpy array: the distance matrix

    """
    n = int(len(features))
    distance_matrix = np.empty([n, n])
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.linalg.norm(features[i] - features[j])

    return distance_matrix

def find_min_overall(forest, distance_matrix):
    """
    Find overall minimum of individual minimum distances.

    Parameters:
    forest (list): list of lists, representing current clusters
    distance_matrix (numpy array): contains distances between all vectors

    Returns:
    dict: dictionary containing information about overall minimum - vectors 
    forming the minimum, the minimum distance, and the size of the new cluster 
    to be formed

    """
    intercluster = []
    for i in range(len(forest)):
        if (forest[i][0] >= 0):
            intercluster.append(find_min_point(i, forest, distance_matrix))

    return min(intercluster, key=lambda x: x['Distance'])

def find_min_point(i, forest, distance_matrix):
    """
    Find minimum of distances for an individual point.

    Parameters:
    i (int): counter from find_min_overall to keep track of current point
    forest (list): list of lists, representing current clusters
    distance_matrix (numpy array): contains distances between all vectors

    Returns:
    dict: dictionary containing information about individual minimum - vectors 
    forming the minimum, the minimum distance, and the size of the new cluster 
    to be formed

    """
    intracluster = []
    # iterate through each point in both clusters being checked
    for j in range(len(forest)):
        for k in range(len(forest[i])):
            for l in range(len(forest[j])):
                if ((forest[i][k] >= 0) & (forest[j][l] >= 0)):
                    if (i != j):  # distance to itself always 0
                        clusterDict = {'P1': i, 'P2': j, 'Distance': distance_matrix[forest[i][k]][forest[j][l]], 'Number': (
                            len(forest[i]) + len(forest[j]))}
                        intracluster.append(clusterDict)

    return min(intracluster, key=lambda x: x['Distance'])

def imshow_hac(Z):
    """
    Display dendrogram visualizing hierarchical clustering

    Parameters:
    Z (numpy array): the linkage matrix representing Single Linkage 
    hierarchical clustering of dataset

    """
    DG(Z)
    plt.show()

if __name__ == '__main__':

    if (len(sys.argv) < 4):
        print('Incorrect Arguments.')
        print('Must include CSV file.')
        print('Must indicate number of rows in CSV file to use. If all rows desired, enter "-1".')
        print('Must indicate features to use. If all features desired, enter "all".')
        print('Usage:', os.path.basename(__file__), '<csvfile>',
              '<number of CSV rows to include>', '<feature 1>', '<...>', '<feature x>')
        sys.exit(1)

    try:
        data = load_data(sys.argv[1])
    except TypeError:
        print('Must provide CSV file to load from.')
        print('Usage:', os.path.basename(__file__), '<csvfile>',
              '<number of CSV rows to include>', '<feature 1>', '<...>', '<feature x>')
        sys.exit(1)
    except FileNotFoundError:
        print('File path does not exist.')
        print('Usage:', os.path.basename(__file__), '<csvfile>',
              '<number of CSV rows to include>', '<feature 1>', '<...>', '<feature x>')
        sys.exit(1)
    except IOError:
        print('File IO Error.')
        print('Usage:', os.path.basename(__file__), '<csvfile>',
              '<number of CSV rows to include>', '<feature 1>', '<...>', '<feature x>')
        sys.exit(1)

    try:
        n = int(sys.argv[2])

        # n cannot be greater or less than the number of rows in CSV file
        if ((n > len(data)) or (n < -1)):
            raise ValueError

        # if n = -1, include all rows in CSV file
        if n == -1:
            n = len(data)

            # check if previous use of isnumeric() is needed

    except ValueError:
        print('Enter number of rows as an integer. Must not exceed number of rows in CSV file or be negative')
        print('except for -1 which indicates use of all CSV rows.')
        print('Usage:', os.path.basename(__file__), '<csvfile>',
              '<number of CSV rows to include>', '<feature 1>', '<...>', '<feature x>')
        sys.exit(1)

    if (sys.argv[3] == 'all'):
        length = len(data[0])
        features = [0] * length
        i = 0
        for key in data[0]:
            if (i < length):
                features[i] = key
                i += 1

    else:
        length = len(sys.argv) - 3
        features = [0] * length
        for i in range(length):
            features[i] = sys.argv[i + 3]

    try:
        Z = single_linkage([filter_features(element, features)
                           for element in data][:n])
    except ValueError:
        print('One or more provided features not found in CSV file fields.')
        print('Usage:', os.path.basename(__file__), '<csvfile>',
              '<number of CSV rows to include>', '<feature 1>', '<...>', '<feature x>')
        sys.exit(1)

    np.save('Z', Z)
    imshow_hac(Z)
