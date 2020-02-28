#
#
#      0===================0
#      |    6 Modelling    |
#      0===================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      First script of the practical session. Plane detection by RANSAC
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points):
    # computes the plane passing through three points
    
    point = points[0].reshape((3, 1)) # ref point 
    normal = np.cross(points[1] - point.T, points[2] - point.T).reshape((3, 1)) # normal of plane
    
    return point, normal / np.linalg.norm(normal)


def in_plane(points, ref_pt, normal, threshold_in=0.1):
    # returns the indices of the points whose distance to the plane are smaller than threshold_in
    
    dists = np.abs((points - ref_pt.T).dot(normal)) # normal component of the vector ref point - query point 
    indices = np.squeeze(dists < threshold_in) # boolean mask
        
    return indices


def RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1):
    
    best_ref_pt = np.zeros((3,1))
    best_normal = np.zeros((3,1))
    best_score = 0
                
    for _ in range(NB_RANDOM_DRAWS):
        # Randomly sample 3 points from the cloud
        sample = points[np.random.choice(len(points), 3, replace = False)]
        # Compute the plane they define
        ref, normal = compute_plane(sample)
        # Count how many points from the cloud are in range of this plane as votes
        indices = in_plane(points, ref, normal, threshold_in)
        
        # The plane that has the most votes is kept as the prominent plane
        if indices.sum() > best_score:
            best_score = indices.sum()
            best_ref_pt = ref
            best_normal = normal 
                
    return best_ref_pt, best_normal


def multi_RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1, NB_PLANES=2):
    
    N = len(points)
    plane_inds = np.zeros(N).astype(bool) # boolean mask (True if the point is assigned to a plane)
    plane_labels = np.zeros(N).astype(int) # labels of corresponding plane for each point 
    remaining_inds = np.arange(N).astype(int) # indices of points not assigned yet

    for i in range(NB_PLANES): 
        # apply RANSAC
        best_ref_pt, best_normal = RANSAC(points[remaining_inds], NB_RANDOM_DRAWS, threshold_in)
        indices = in_plane(points[remaining_inds], best_ref_pt, best_normal, threshold_in)
        
        # updates
        plane_inds[remaining_inds[indices]] = True
        plane_labels[remaining_inds[indices]] = i + 1
        remaining_inds = remaining_inds[indices == False]
        
    return plane_inds, remaining_inds, plane_labels


# ----------------------------------------------------------------------------------------------------------------------
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
    N = len(points)

    # Computes the plane passing through 3 randomly chosen points
    # ***********************************************************
    #

    if False:

        # Define parameter
        threshold_in = 0.1

        # Take randomly three points
        pts = points[np.random.randint(0, N, size=3)]

        # Computes the plane passing through the 3 points
        t0 = time.time()
        ref_pt, normal = compute_plane(pts)
        t1 = time.time()
        print('plane computation done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        t0 = time.time()
        points_in_plane = in_plane(points, ref_pt, normal, threshold_in)
        t1 = time.time()
        print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
        plane_inds = points_in_plane.nonzero()[0]

        # Save the 3 points and their corresponding plane for verification
        pts_clr = np.zeros_like(pts)
        pts_clr[:, 0] = 1.0
        write_ply('../triplet.ply',
                  [pts, pts_clr],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../triplet_plane.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Computes the best plane fitting the point cloud
    # ***********************************
    #

    if False:

        # Define parameters of RANSAC
        NB_RANDOM_DRAWS = 100
        threshold_in = 0.05

        # Find best plane by RANSAC
        t0 = time.time()
        best_ref_pt, best_normal = RANSAC(points, NB_RANDOM_DRAWS, threshold_in)
        t1 = time.time()
        print('RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        points_in_plane = in_plane(points, best_ref_pt, best_normal, threshold_in)
        plane_inds = points_in_plane.nonzero()[0]
        remaining_inds = (1-points_in_plane).nonzero()[0]

        # Save the best extracted plane and remaining points
        write_ply('../best_plane.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../remaining_points.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Find multiple planes in the cloud
    # *********************************
    #

    if True:

        # Define parameters of multi_RANSAC
        NB_RANDOM_DRAWS = 200
        threshold_in = 0.05
        NB_PLANES = 5

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = multi_RANSAC(points, NB_RANDOM_DRAWS, threshold_in, NB_PLANES)
        t1 = time.time()
        print('\nmulti RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply('../best_planes.ply',
                  [points[plane_inds], colors[plane_inds], plane_labels[plane_inds].astype(np.int32)],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])
        write_ply('../remaining_points_.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

        print('Done')
