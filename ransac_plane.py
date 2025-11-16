"""
Create: 2022.06.15
Author: SG.SUH
Python: 3.7
"""

import numpy as np

from least_square import least_square

def ransac_plane(pc, rand_no, iter_no, in_dist_thre, in_no):
    assert pc.shape[1] == 3, 'The input point must be n * 3'
    assert rand_no >= 3, 'At least 3 points are needed to estimate a plane'

    success = 0
    error_min = np.inf

    while success == 0:
        iterations = 0

        while iterations <= iter_no:
            sample = np.random.randint(rand_no, size = pc.shape[0])
            in_pc = pc[sample]

            norm_plane, dist_plane = least_square(in_pc)

            dist_pc2plane = pc[:, 0:3] * norm_plane - dist_plane
            dist_pc2plane = np.abs(dist_pc2plane)

            ind_1 = np.argwhere(dist_pc2plane < in_dist_thre)

            len1 = ind_1.shape[0]

            if len1 > in_no:
                in_pc = pc[ind_1]

                norm_plane, dist_plane = least_square(in_pc)

                dist_pc2plane = pc[:, :] * norm_plane - dist_plane
                dist_pc2plane = np.abs(dist_pc2plane)

                ind_2 = np.argwhere(dist_pc2plane < in_dist_thre)

                error = np.sum(dist_pc2plane[ind_2]) / ind_2.shape[0]

                len2 = ind_1.shape[0]

                if len2 > in_no:
                    in_pc = pc[ind_2]

                    norm_plane, dist_plane = least_square(in_pc)

                    dist_pc2plane = pc[:, :] * norm_plane - dist_plane
                    dist_pc2plane = np.abs(dist_pc2plane)
                    
                    ind_2 = np.argwhere(dist_pc2plane < in_dist_thre)

                    error = np.sum(dist_pc2plane[ind_2]) / ind_2.shape[0]
                else:
                    error = np.inf
            else:
                error = np.inf
            
            if error < error_min:
                error_min = error
                plane_norm_r = norm_plane
                dist_plane_r = dist_plane
                pc_in_r = in_pc

            if error_min != np.inf:
                success = 1

            iterations += 1

    return plane_norm_r, dist_plane_r
