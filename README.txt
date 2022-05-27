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
