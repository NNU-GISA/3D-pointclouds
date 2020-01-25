#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
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
from utils.visu import show_ICP


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # YOUR CODE
    pm_data, pm_ref = data.mean(1).reshape((-1, 1)), ref.mean(1).reshape((-1, 1))
    Q_data, Q_ref = data - pm_data, ref - pm_ref
    H = Q_data@(Q_ref.T)
    U, S, VT = np.linalg.svd(H)

    R = (U.dot(VT)).T
    T = pm_ref - R.dot(pm_data)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = (U.dot(VT)).T
        T = pm_ref - R.dot(pm_data)
    return R, T

def RMS(pts_1, pts_2):
    total_sqrms = np.power(np.linalg.norm(pts_1 - pts_2, axis = 0), 2)
    rms = np.sqrt(total_sqrms.sum() / pts_1.shape[1])

    return rms

def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []

    # YOUR CODE
    rms_list = []

    ref_tree = KDTree(ref.T, leaf_size = 2) 
    rms, it = np.inf, 0

    while it < max_iter and rms > RMS_threshold:
        matching_idx = ref_tree.query(data_aligned.T, k = 1, return_distance = False)
        matching_pts = ref[:, np.squeeze(matching_idx)]
        R, T = best_rigid_transform(data_aligned, matching_pts)

        if it == 0:
            R_list.append(R)
            T_list.append(T)
        else:
            R_list.append(R.dot(R_list[-1]))
            T_list.append(R.dot(T_list[-1]) + T)

        neighbors_list.append(matching_idx.reshape((-1)))
        data_aligned = R.dot(data_aligned) + T
        rms = RMS(data_aligned, matching_pts)
        rms_list.append(rms)
        it += 1

    return data_aligned, R_list, T_list, neighbors_list, rms_list

def icp_point_to_point_stochastic(data, ref, max_iter, RMS_threshold, sampling_limit, final_overlap = 1., partial = 1.):

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    rms_list = []

    ref_tree = KDTree(ref.T, leaf_size = 2) 
    rms, it, num_pts = np.inf, 0, data_aligned.shape[1]

    while it < max_iter and rms > RMS_threshold:
        idx = np.random.choice(num_pts, sampling_limit, replace = False)
        data_aligned_sub = data_aligned[:, idx]
        matching_dist, matching_idx = ref_tree.query(data_aligned_sub.T, k = 1)

        if final_overlap < 1:
            keep_num = int(final_overlap * data_aligned_sub.shape[1])
            keep_idx = np.argsort(np.squeeze(matching_dist))[:keep_num]
            matching_idx = matching_idx[keep_idx]
            data_aligned_sub = data_aligned_sub[:, keep_idx]

        matching_pts = ref[:, np.squeeze(matching_idx)]
        R, T = best_rigid_transform(data_aligned_sub, matching_pts)

        if it == 0:
            R_list.append(R)
            T_list.append(T)
        else:
            R_list.append(R.dot(R_list[-1]))
            T_list.append(R.dot(T_list[-1]) + T)

        neighbors_list.append(matching_idx.reshape((-1)))
        data_aligned = R.dot(data_aligned) + T
        if partial < 1:
            rms = RMS_partial(data_aligned_sub, matching_pts, overlap = partial)
            rms_list.append(rms)
        else:
            rms = RMS(data_aligned_sub, matching_pts)
            rms_list.append(rms)
        it += 1
    return data_aligned, R_list, T_list, neighbors_list, rms_list

def icp_point_to_point_stochastic_partial(data, ref, max_iter, RMS_threshold, sampling_limit, final_overlap = 1., partial = [0.5,0.7,0.9, 1.]):

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    rms_lists = dict.fromkeys(np.arange(len(partial)))
    for i in range(len(partial)):
        rms_lists[i] = []

    ref_tree = KDTree(ref.T, leaf_size = 2) 
    rms, it, num_pts = np.inf, 0, data_aligned.shape[1]

    while it < max_iter and rms > RMS_threshold:
        idx = np.random.choice(num_pts, sampling_limit, replace = False)
        data_aligned_sub = data_aligned[:, idx]
        matching_dist, matching_idx = ref_tree.query(data_aligned_sub.T, k = 1)

        if final_overlap < 1:
            keep_num = int(final_overlap * data_aligned_sub.shape[1])
            keep_idx = np.argsort(np.squeeze(matching_dist))[:keep_num]
            matching_idx = matching_idx[keep_idx]
            data_aligned_sub = data_aligned_sub[:, keep_idx]

        matching_pts = ref[:, np.squeeze(matching_idx)]
        R, T = best_rigid_transform(data_aligned_sub, matching_pts)

        if it == 0:
            R_list.append(R)
            T_list.append(T)
        else:
            R_list.append(R.dot(R_list[-1]))
            T_list.append(R.dot(T_list[-1]) + T)

        neighbors_list.append(matching_idx.reshape((-1)))
        data_aligned = R.dot(data_aligned) + T
        for i in range(len(partial)):
            rms = RMS_partial(data_aligned_sub, matching_pts, overlap = partial[i])
            rms_lists[i].append(rms)
        it += 1
    return data_aligned, R_list, T_list, neighbors_list, rms_lists

def RMS_partial(pts_1, pts_2, overlap = 1.):

    total_sqrms = np.power(np.linalg.norm(pts_1 - pts_2, axis = 0), 2)
    if overlap < 1.:
        keep_num = int(pts_1.shape[1] * overlap)
        total_sqrms = np.sort(total_sqrms)[:keep_num]

    rms = np.sqrt(total_sqrms.sum() / len(total_sqrms))

    return rms


#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

        # Load clouds
        bunny_original_file = read_ply(bunny_o_path)
        bunny_returned_file = read_ply(bunny_r_path)

        bunny_original_pts = np.vstack((bunny_original_file['x'], bunny_original_file['y'], bunny_original_file['z'])) 
        bunny_returned_pts = np.vstack((bunny_returned_file['x'], bunny_returned_file['y'], bunny_returned_file['z'])) 

        # Find the best transformation
        R, T = best_rigid_transform(bunny_original_pts, bunny_returned_pts)

        # Apply the tranformation
        bunny_transformed_pts = R.dot(bunny_original_pts) + T

        # Save cloud
        write_ply('../bunny_transform.ply', [bunny_transformed_pts.T], ['x', 'y', 'z'])

        # Compute RMS
        rms = RMS(bunny_transformed_pts, bunny_returned_pts)

        # Print RMS
        print("Final RMS: %f" % rms)
   

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'

        # Load clouds
        ref2D_file = read_ply(ref2D_path)
        data2D_file = read_ply(data2D_path)
        
        ref2D_pts = np.vstack((ref2D_file['x'], ref2D_file['y'])) # data
        data2D_pts = np.vstack((data2D_file['x'], data2D_file['y'])) # ref

        # Apply ICP
        max_iter = 30
        RMS_threshold = 0.06
        data_aligned, R_list, T_list, neighbors_list, rms_list = icp_point_to_point(data2D_pts, ref2D_pts, max_iter, RMS_threshold)

        # Show ICP
        show_ICP(data2D_pts, ref2D_pts, R_list, T_list, neighbors_list)
        plt.style.use('seaborn')
        plt.plot(rms_list)
        plt.xlabel("iteration")
        plt.ylabel("RMS")
        plt.show()

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'

        # Load clouds
        bunny_original_file = read_ply(bunny_o_path)
        bunny_perturbed_file = read_ply(bunny_p_path)

        bunny_original_pts = np.vstack((bunny_original_file['x'], bunny_original_file['y'], bunny_original_file['z'])) 
        bunny_perturbed_pts = np.vstack((bunny_perturbed_file['x'], bunny_perturbed_file['y'], bunny_perturbed_file['z'])) 

        # Apply ICP
        max_iter = 30
        RMS_threshold = 1e-4
        data_aligned, R_list, T_list, neighbors_list, rms_list = icp_point_to_point(bunny_perturbed_pts, bunny_original_pts, max_iter, RMS_threshold)

        # Show ICP
        show_ICP(bunny_perturbed_pts, bunny_original_pts, R_list, T_list, neighbors_list)
        plt.plot(rms_list)
        plt.xlabel("iteration")
        plt.ylabel("RMS")
        plt.show()


    # Fast ICP
    # ********
    #

    # If statement to skip this part if wanted
    if True:

        # Cloud paths
        NDDC_1_path = '../data/Notre_Dame_Des_Champs_1.ply'
        NDDC_2_path = '../data/Notre_Dame_Des_Champs_2.ply'

        # Load clouds
        NDDC_1_file= read_ply(NDDC_1_path)
        NDDC_2_file = read_ply(NDDC_2_path)

        NDDC_1_pts = np.vstack((NDDC_1_file['x'], NDDC_1_file['y'], NDDC_1_file['z'])) 
        NDDC_2_pts = np.vstack((NDDC_2_file['x'], NDDC_2_file['y'], NDDC_2_file['z'])) 

        # Apply fast ICP for different values of the sampling_limit parameter
        max_iter = 100
        RMS_threshold = 1e-4
        # sampling_limits = [1000,10000,50000]
        overlaps = [0.3,0.5,0.7,1.]
        # for lim in sampling_limits:
        rms_lists = []
        for overlap in overlaps:
            NDDC_aligned, R_list, T_list, neighbors_list, rms_dic = icp_point_to_point_stochastic_partial(NDDC_2_pts, NDDC_1_pts, max_iter, RMS_threshold, 10000, overlap)
            rms_lists.append(rms_dic)
            
        # Plot RMS
        #
        # => To plot something in python use the function plt.plot() to create the figure and 
        #    then plt.show() to display it
        write_ply('../NDDC_transform.ply', [NDDC_aligned.T], ['x', 'y', 'z']) 
        for i, overlap in enumerate(overlaps):
            for j, ov in enumerate([0.5,0.7,0.9, 1.]):
                plt.plot(rms_lists[i][j], label = "Overlap"+str(overlap)+", RMS Overlap"+str(ov))
        plt.legend()
        plt.xlabel("iteration")
        plt.ylabel("RMS")
        plt.show()
