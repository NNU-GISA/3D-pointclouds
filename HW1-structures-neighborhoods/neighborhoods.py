#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time

import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):

    # YOUR CODE
    neighborhoods = []
    for query in queries:
        idx = np.where(np.linalg.norm(supports - query, axis = 1) < radius)[0]
        neighborhoods.append(supports[idx])

    return neighborhoods


def brute_force_KNN(queries, supports, k):

    # YOUR CODE
    neighborhoods = []
    for query in queries:
        idx = np.argsort(np.linalg.norm(supports - query))[:k]
        neighborhoods.append(supports[idx])

    return neighborhoods

class hierarchical_search:
    
    def __init__(self, points, leaf_size):

        self.tree = KDTree(points, leaf_size = leaf_size)
        self.points = points

    def query_radius(self, queries, radius):

        idx = self.tree.query_radius(queries, radius)
        neighborhoods = [self.points[i] for i in idx]

        return neighborhoods
    
# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if False:

        # Define the search parameters
        neighbors_num = 1000
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))



    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if True:

        # Define the search parameters
        # num_queries = 1000
        num_queries = 1000

        # YOUR CODE
        # radius = 0.2
        radius = [0.1,0.2,0.4,0.6,0.8,1,1.2,1.4]
        # leaf_sizes = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
        leaf_size = 25
        query_times = []
        # Draw random points
        idx = np.random.choice(points.shape[0], num_queries, replace=False)
        
        #for leaf_size in leaf_sizes:
        for r in radius:

            # Create tree
            kdtree = hierarchical_search(points, leaf_size = leaf_size)

            # Queries
            queries = points[idx, :]

            # Search spherical neighbors
            t0 = time.time()
            neighborhoods = kdtree.query_radius(queries, r)
            t1 = time.time()        
            # print("leaf size:", leaf_size)
            print("radius:", r)
            print('{:d} spherical neighborhood computed in {:.3f} seconds'.format(num_queries, t1 - t0))

            query_times.append(t1 - t0)
        
        plt.style.use('seaborn')
        plt.scatter(radius, query_times)
        plt.xlabel("Radius (m)")
        plt.ylabel("Query Times (s)")
        plt.show()
        
        

        