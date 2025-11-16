"""
Create: 2022.06.15
Author: SG.SUH
Python: 3.7
"""

import numpy as np

def least_square(p):
    pbar = np.mean(p)

    asum = p[:, :] - pbar
    asum = np.cov(asum)

    w, V = np.linalg.eig(asum)

    n_est = V[:, 0]

    ro_est = np.dot(n_est, pbar)

    return n_est, ro_est