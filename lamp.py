import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KDTree


# Forced projection method as decribed in the paper "On improved projection 
# techniques to support visual exploration of multi-dimensional data sets"
def force_method(X, init='random', delta_frac=10.0, max_iter=50):
    # TODO: tSNE from scikit learn can initialize projections based on PCA
    if init == 'random':
        X_proj = np.random.rand(X.shape[0], 2)
    # TODO: something like:
    # else if init == ' PCA':
    #   X_proj = PCA(X)

    vec_dist = distance.pdist(X, 'sqeuclidean')
    dmin = np.min(vec_dist)
    dmax = np.max(vec_dist)
    dist_matrix = distance.squareform(vec_dist)

    dist_diff = dmax - dmin

    # TODO: better stopping criteria?
    # TODO: this is _slow_: consider using squared distances when possible
    #       - using sqeuclidean it is faster but results are worse
    for k in range(max_iter):
        for i in range(X_proj.shape[0]):
            x_prime = X_proj[i]
            for j in range(X_proj.shape[0]):
                if i == j:
                    # FIXME: the paper compares x\prime to q\prime, here I'm
                    # comparing only the indices
                    continue
                q_prime = X_proj[j]

                #if np.allclose(x_prime, q_prime):
                #    continue

                v = q_prime - x_prime
                dist_xq = distance.sqeuclidean(x_prime, q_prime)
                delta = (dist_matrix[i, j] - dmin)/dist_diff - dist_xq
                # FIXME the algorithm desbribed in the paper states:
                # "move q_prime in the direction of v by a fraction of delta"
                # what is a good value for delta_frac?
                delta /= delta_frac

                X_proj[j] = X_proj[j] + v*delta

    # TODO: is normalization really necessary?
    X_proj = (X_proj - X_proj.min(axis=0)) / (X_proj.max(axis=0) - X_proj.min(axis=0))

    return X_proj

# Heavily based on lamp implementation from: 
# https://github.com/thiagohenriquef/mppy

# In my tests, this method worked reasonably well when data was normalized
# in range [0,1]. 
def lamp2d(X, ctrl_pts_idx=None):
    # k: the number of control points
    # LAMP paper argues that few control points are needed. sqrt(|X|) is used
    # here as it the necessary number for other methods
    if ctrl_pts_idx is None:
        k = int(np.sqrt(X.shape[0]))
        ctrl_pts_idx = np.random.randint(0, X.shape[0], k)

    X_s = X[ctrl_pts_idx]
    Y_s = force_method(X_s)

    X_proj = np.zeros((X.shape[0], 2))
    # LAMP algorithm
    for idx in range(X.shape[0]):
        skip = False

        # 1. compute weighs alpha_i
        alpha = np.zeros(X_s.shape[0])
        for i in range(X_s.shape[0]):
            diff = X_s[i] - X[idx] 
            diff2 = np.dot(diff, diff)
            if diff2 < 1e-4:
                # X_s[i] and X[idx] are almost the same point, so
                # project to the same point (Y_s[i]
                X_proj[idx] = Y_s[i]
                skip = True
                break
            alpha[i] = 1.0/diff2

        if skip == True:
            continue

        # 2. compute x_tilde, y_tilde
        sum_alpha = np.sum(alpha)
        x_tilde = np.sum(alpha[:, np.newaxis]*X_s, axis=0)/sum_alpha
        y_tilde = np.sum(alpha[:, np.newaxis]*Y_s, axis=0)/sum_alpha

        # 3. build matrices A and B
        x_hat = X_s - x_tilde
        y_hat = Y_s - y_tilde
        
        alpha_sqrt = np.sqrt(alpha)
        A = alpha_sqrt[:, np.newaxis]*x_hat
        B = alpha_sqrt[:, np.newaxis]*y_hat

        # 4. compute the SVD decomposition UDV from (A^T)B
        u, s, vh = np.linalg.svd(np.dot(A.T, B))
        # 5. Make M = UV

        aux = np.zeros((X.shape[1], 2))
        aux[0] = vh[0]
        aux[1] = vh[1]
        M = np.dot(u, aux)
        # 6. Compute the mapping (x - x_tilde)M + y_tilde
        X_proj[idx] = np.dot(X[idx] - x_tilde, M) + y_tilde

    return X_proj

def ilamp(data, data_proj, p, k=6):
    # 0. compute X_s and Y_s
    tree = KDTree(data_proj)

    dist, ind = tree.query([p], k=k)
    # ind is a (1xdim) array
    ind = ind[0]
    X_proj = data_proj[ind]
    X = data[ind]

    # 1. compute weights alpha_i
    alpha = np.zeros(X_proj.shape[0])
    for i in range(X_proj.shape[0]):
        diff = X_proj[i] - p
        diff2 = np.dot(diff, diff)

        if diff2 < 1e-4:
            # difference is too small, the counter part to p
            # precisely X[i]
            return X[i]
        alpha[i] = 1.0/diff2

    sum_alpha = np.sum(alpha)
    # 2. compute x_tilde, y_tilde
    x_tilde = np.sum(alpha[:, np.newaxis]*X, axis=0)/sum_alpha
    y_tilde = np.sum(alpha[:, np.newaxis]*X_proj, axis=0)/sum_alpha

    # 3. build matrices A and B
    x_hat = X - x_tilde
    y_hat = X_proj - y_tilde

    alpha_sqrt = np.sqrt(alpha)
    A = alpha_sqrt[:, np.newaxis]*y_hat
    B = alpha_sqrt[:, np.newaxis]*x_hat

    u, s, vh = np.linalg.svd(np.dot(A.T, B))

    # 5. let M = UV
    aux = np.zeros((2, X.shape[1]))
    aux[0] = vh[0]
    aux[1] = vh[1]
    M = np.dot(u, aux)

    return np.dot(p - y_tilde, M) + x_tilde

