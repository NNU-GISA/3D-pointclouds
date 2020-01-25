#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      First script of the practical session. Transformation of a point cloud
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

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#

def center_points(pts):
    centroid = pts.mean(0).reshape((1, -1))
    centered_pts = pts - centroid
    return centered_pts, centroid

def scale_points(pts, factor=2):
    return(pts/factor)

def rotate_points(pts, axis = 2, angle = -90):
    unit_vec = np.zeros((3, 1))
    unit_vec[axis] = 1.
    cos_angle, sin_angle = np.cos(angle / 180 * np.pi), np.sin(angle / 180 * np.pi)

    trans1 = (1 - cos_angle) * unit_vec.dot(unit_vec.T)
    trans2 = sin_angle * np.array(([[0, -unit_vec[2, 0], unit_vec[1, 0]],
                                        [unit_vec[2, 0], 0, -unit_vec[0, 0]],
                                        [-unit_vec[1, 0], unit_vec[0, 0], 0]]))
    trans3 = np.eye(3) * cos_angle
    trans = trans1 + trans2 + trans3

    return pts.dot(trans.T)

def translate_points(pts, a):
    return(pts+np.array(a).reshape(1,3))

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

    # Path of the file
    file_path = '../data/bunny.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate x, y, and z in a (N*3) point matrix
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Concatenate R, G, and B channels in a (N*3) color matrix
    colors = np.vstack((data['red'], data['green'], data['blue'])).T

    # Get the scalar field which represent density as a vector
    density = data['scalar_density']

    # Transform point cloud
    # *********************
    #
    #   Follow the instructions step by step
    #
    # Replace this line by your code
    transformed_points, centroid = center_points(points)
    transformed_points = scale_points(transformed_points)
    transformed_points = rotate_points(transformed_points)
    transformed_points = transformed_points + centroid
    transformed_points = translate_points(transformed_points,[0, -0.1, 0])
    
    # Save point cloud
    # *********************
    #
    #   Save your result file
    #   (See write_ply function)
    #

    # Save point cloud
    write_ply('../little_bunny.ply', [transformed_points, colors, density], ['x', 'y', 'z', 'red', 'green', 'blue', 'density'])

    print('Done')
