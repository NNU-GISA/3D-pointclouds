#
#
#      0===================0
#      |    6 Modelling    |
#      0===================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Plane detection by region growing
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

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

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

# Cf TP3
def local_PCA(points):
    
    centroid = points.mean(0).reshape((1, -1))
    centered_pts = points - centroid
    cov = centered_pts.T@centered_pts/len(centered_pts)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = np.flip(eigenvectors, 1)

    return eigenvalues, eigenvectors

# Cf TP3
def neighborhood_PCA(query_points, cloud_points, radius):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    
    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    tree = KDTree(cloud_points)
    l = tree.query_radius(query_points, radius)

    for i, idx in enumerate(l):
        pts = cloud_points[idx]
        v, w = local_PCA(pts)
        all_eigenvalues[i] = v
        all_eigenvectors[i] = w

    return all_eigenvalues, all_eigenvectors

def compute_planarities_and_normals(points, radius):
    # computes for each point of the planarity and the normal with the radius r

    all_eigenvalues, all_eigenvectors = neighborhood_PCA(points, points, radius)

    normals = all_eigenvectors[:, :, -1]
    planarities = (all_eigenvalues[:, 1] - all_eigenvalues[:, 2]) / np.clip(all_eigenvalues[:, 0], 1e-6, None)

    return planarities, normals

def region_criterion(p1, p2, n1, n2, threshold1 = 0.1, threshold2 = 10):
    # returns True if two conditions are met:
# -	distance from the point p2 to the plane (p1, n1) is smaller than threshold1
# -	normals n1 and n2 form an angle smaller than threshold2

    dist_plane = np.abs(n1.dot(p2- p1))
    angle = 180 * np.arccos(np.abs(np.clip(n1.dot(n2), -1., 1.))) / np.pi

    return True if dist_plane < threshold1 and angle < threshold2 else False


def queue_criterion(p, threshold = 0.8):
    # that returns True if planarity p is bigger than a certain threshold
    return True if p > threshold else False


def RegionGrowing(cloud, normals, planarities, radius):

    N = len(cloud)
    region = np.zeros(N, dtype=bool)

    # Choose seed and instantiate region
    queue = [np.argmax(planarities)]
    region[queue[0]] = True

    # Get all neighbours
    tree = KDTree(cloud, 10)
    neigh = tree.query_radius(cloud, radius)

    while queue:
        # Extract a point and get its neighbours
        q = queue.pop(0)
        neighq = [ind for ind in neigh[q] if not region[ind]]
        for p in neighq:
            reg_criterion = region_criterion(cloud[q], cloud[p], normals[q], normals[p])
            # Add to region if meets region criterion, and queue if neighbour meets queue criterion
            if reg_criterion:
                region[p] = True
                if queue_criterion(planarities[p]):
                    queue.append(p)

    return region


def multi_RegionGrowing(cloud, normals, planarities, radius, NB_PLANES=2):

    N = len(cloud)
    plane_inds = np.zeros((N,)).astype(bool)
    plane_labels = np.zeros((N,)).astype(int)
    remaining_inds = np.arange(N).astype(int)

    for i in range(NB_PLANES):
        best_region = RegionGrowing( cloud[remaining_inds], 
                                     normals[remaining_inds], 
                                     planarities[remaining_inds],
                                     radius )
        plane_inds[remaining_inds[best_region]] = True
        plane_labels[remaining_inds[best_region]] = i + 1
        remaining_inds = remaining_inds[best_region == False]

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

    # Computes normals of the whole cloud
    # ***********************************
    #

    # Parameters for normals computation
    radius = 0.2

    # Computes normals of the whole cloud
    t0 = time.time()
    planarities, normals = compute_planarities_and_normals(points, radius)
    t1 = time.time()
    print('normals and planarities computation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../planarities.ply',
              [points, planarities],
              ['x', 'y', 'z', 'planarities'])

    # Find a plane by Region Growing
    # ******************************
    #

    if True:
        # Define parameters of Region Growing
        radius = 0.2

        # Find a plane by Region Growing
        t0 = time.time()
        region = RegionGrowing(points, normals, planarities, radius)
        t1 = time.time()
        print('Region Growing done in {:.3f} seconds'.format(t1 - t0))

        # Get inds from bollean array
        plane_inds = region.nonzero()[0]
        remaining_inds = (1 - region).nonzero()[0]

        # Save the best plane
        write_ply('../best_plane.ply',
                  [points[plane_inds], colors[plane_inds], planarities[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'planarities'])
        write_ply('../remaining_points.ply',
                  [points[remaining_inds], colors[remaining_inds], planarities[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'planarities'])

    # Find multiple in the cloud
    # ******************************
    #

    if True:
        # Define parameters of multi_RANSAC
        radius = 0.2
        NB_PLANES = 10

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = multi_RegionGrowing(points, normals, planarities, radius, NB_PLANES)
        t1 = time.time()
        print('multi RegionGrowing done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply('../best_planes.ply',
                  [points[plane_inds], colors[plane_inds], plane_labels[plane_inds].astype(np.int32)],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])
        write_ply('../remaining_points_.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

        print('Done')
