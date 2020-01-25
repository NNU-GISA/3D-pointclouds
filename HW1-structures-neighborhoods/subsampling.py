#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
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
from sklearn.preprocessing import label_binarize

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time

from collections import Counter

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def cloud_decimation(points, colors, labels, factor):

    # YOUR CODE
    idx = np.arange(0, len(points), factor)
    decimated_points = points[idx]
    decimated_colors = colors[idx]
    decimated_labels = labels[idx]

    return decimated_points, decimated_colors, decimated_labels


def grid_subsampling(points, voxel_size):

    #   Tips :
    #       > First compute voxel indices in each direction for all the points (can be negative).
    #       > Sum and count the points in each voxel (use a dictionaries with the indices as key).
    #         Remember arrays cannot be dictionary keys, but tuple can.
    #       > Divide the sum by the number of point in each cell.
    #       > Do not forget you need to return a numpy array and not a dictionary.
    #

    # YOUR CODE
    mini = points.min(0)
    idx = np.floor((points - mini) / voxel_size)
    _, pts_grids = np.unique(idx, return_inverse = True, axis = 0)
    num_grids = pts_grids.max() + 1 

    subsampled_points = []
    for i in range(num_grids):
        subsampled_points.append(points[pts_grids == i].mean(0))

    return np.array(subsampled_points)


def grid_subsampling_colors(points, colors, labels, voxel_size):

    # YOUR CODE
    mini = points.min(0)
    idx = np.floor((points - mini) / voxel_size)
    _, pts_grids = np.unique(idx, return_inverse = True, axis = 0)
    
    subsampled_points = grid_subsampling(points, voxel_size)
    subsampled_colors = []
    subsampled_labels = []
    for i in range(len(subsampled_points)):
        subsampled_colors.append(colors[pts_grids == i].astype(float).mean(0).astype(np.uint8))
        mode = Counter(labels[pts_grids == i]).most_common(1)
        subsampled_labels.append(mode[0][0])

    return subsampled_points, np.array(subsampled_colors), np.array(subsampled_labels)


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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']    

    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 300

    # Decimate
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
    t1 = time.time()
    print('decimation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    # Subsample the point cloud on a grid
    # ***********************************
    #

    # Define the size of the grid
    voxel_size = 0.2

    # Subsample
    t0 = time.time()
    subsampled_points, subsampled_colors, subsampled_labels = grid_subsampling_colors(points, colors, labels, voxel_size)
    t1 = time.time()
    print('Subsampling done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../grid_subsampled.ply', [subsampled_points,subsampled_colors, subsampled_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    print('Done')
