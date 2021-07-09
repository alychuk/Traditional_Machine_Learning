import numpy as np

X = np.array([ [2.5, 2.4], [.5, .7], [2.2, 2.9], [1.9, 2.2],
                [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1],
                [1.5, 1.6], [1.1,.9] ])


def compute_Z(X, centering=True, scaling=False):
    col_means = np.mean(X, axis=0)
    for i in range(np.size(X[0])):
        X[:, i] = X[:, i] - col_means[i]
    return X


def compute_covariance_matrix(Z):
    return Z.T @ Z


def find_pcs(COV):
    L, PCS = np.linalg.eig(COV)
    new = np.vstack((L, PCS))
    sorted = new[:, new[0].argsort()[::-1]]
    L = sorted[0,:]
    PCS = sorted[1:]
    return PCS, L


def project_data(Z, PCS, L, k, var):
    if var == 0:
        if k <= L.size:
            Z_star = Z @ PCS[...,:k]
    if k == 0:
        print("placeholder")
    return Z_star

Z = compute_Z(X)
COV = compute_covariance_matrix(Z)
PCS, L =find_pcs(COV)
Z_star = project_data(Z, PCS, L, 2, 0)
print(COV)
print(Z_star)
