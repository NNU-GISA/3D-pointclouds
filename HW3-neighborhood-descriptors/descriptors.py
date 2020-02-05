#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
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

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#

def local_PCA(points):
    
    centroid = points.mean(0).reshape((1, -1))
    centered_pts = points - centroid
    cov = centered_pts.T@centered_pts/len(centered_pts)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = np.flip(eigenvectors, 1)

    return eigenvalues, eigenvectors


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


def compute_features(query_points, cloud_points, radius):

    # Compute the features for all query points in the cloud

    eps = 1e-10
    eigenvalues, eigenvectors = neighborhood_PCA(query_points, cloud_points, radius)
    linearity = 1-eigenvalues[:,1]/np.clip(eigenvalues[:, 0], eps, None)
    planarity = (eigenvalues[:,1] - eigenvalues[:,2])/np.clip(eigenvalues[:, 0], eps, None)
    sphericity = eigenvalues[:,2]/np.clip(eigenvalues[:, 0], eps, None)
    normals = eigenvectors[:, :, -1]
    verticality = 2*np.abs(np.arcsin(normals[:, -1]))/np.pi
    
    return verticality, linearity, planarity, sphericity


def compute_add_features(query_points, cloud_points, radius):

    # Compute the features for all query points in the cloud
    
    eps = 1e-10
    eigenvalues, eigenvectors = neighborhood_PCA(query_points, cloud_points, radius)
    normals = eigenvectors[:, :, -1]
    tangents = eigenvectors[:, :, 0]
    omnivariance = np.sign(eigenvalues[:,0]*eigenvalues[:,1]*eigenvalues[:,2])*(np.abs(eigenvalues[:,0]*eigenvalues[:,1]*eigenvalues[:,2]))**(1/3)
    anisotropy = (eigenvalues[:,0] - eigenvalues[:,2])/np.clip(eigenvalues[:, 0], eps, None)
    eigenentropy = -np.sum(eigenvalues*np.log(np.clip(eigenvalues, eps, None)), axis = 1)
    eigensum = eigenvalues[:,0] + eigenvalues[:,1] + eigenvalues[:,2]
    change_curvature = eigenvalues[:,2]/np.clip(eigensum, eps, None)
    
    angles_normals = normals@[1,0,0]
    angles_tangents = tangents@[0,0,1]
    return(np.cos(angles_normals),np.sin(angles_normals), np.cos(angles_tangents), np.sin(angles_tangents), omnivariance, anisotropy, eigenentropy, change_curvature)


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = local_PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

    # Normal computation
    # ******************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # YOUR CODE
        radius = 1
        eigenvalues, eigenvectors = neighborhood_PCA(cloud, cloud, radius) 
        normals = eigenvectors[:, :, -1]
        write_ply('Lille_small_normals.ply', [cloud, normals], ['x', 'y', 'z', 'nx', 'ny', 'nz'])

    # Features computation
    # ********************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # YOUR CODE
        radius = 1
        features = compute_features(cloud, cloud, radius)
        write_ply('Lille_small_features.ply', [cloud] + list(features), ['x', 'y', 'z', 'verticality', 'linearity', 'planarity', 'sphericity'])
