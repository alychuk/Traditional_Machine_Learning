import numpy as np
import random

def K_Means(X,K):
    """ Performs the K-means algorithm """
    max_val = np.max(X, axis=0)
    min_val = np.min(X, axis=0)

    num_samples = X.shape[0]
    num_features = X.shape[1]

    # Assigns each sample to a random class
    classes = np.random.randint(low=0, high=K, size=num_samples)
    # Initializes list of centers with random values between the min and max
    centers = np.random.uniform(low=0., high=1., size=(K, num_features))
    centers = centers * (max_val - min_val) + min_val

    # K-means for 100 iterations in the hopes of convergence
    for i in range(0,100):
        # Finds the distance between cluster centers and sample points
        distances = np.array(
            [np.linalg.norm(X - c, axis=1) for c in centers])
        # Assigns points to classes based on the center they're closest too
        new_classes = np.argmin(distances, axis=0)

        if (classes == new_classes).all():
            # If no points have been reassigned, k-means converged
            classes = new_classes
            break
        else:
            classes = new_classes
            for C in range(K):
                # compute new centers by taking the mean of points in the centers class
                centers[C] = np.mean(X[classes == C], axis=0)
    return centers


def K_Means_better(X,K):
    cluster_count = 0
    possible_centers = []
    prev_center_set = K_Means(X,K)
    for i in range(0,1000):
        new_center_set = K_Means(X,K)
        if (np.array_equal(prev_center_set,new_center_set)):
            return prev_center_set

"""
def K_Means_better(X,K):
    cluster_count = 0
    possible_centers = []
    prev_center_set = K_Means(X,K)
    for i in range(0,1000):
        new_center_set = K_Means(X,K)
        if (np.array_equal(prev_center_set,new_center_set)):
            possible_centers.append(new_center_set)
        if possible_centers:
            for find in range(possible_centers)
"""

"""
X = np.array([[0], [1], [2], [7], [8], [9], [12], [14], [15]])
X2 = np.array([[1, 0], [7, 4], [9, 6], [2, 1], [4, 8], [0, 3], [13, 5], [6, 8], [7, 3], [3, 6], [2, 1], [8, 3], [10, 2], [3, 5], [5, 1], [1, 9], [10, 3], [4, 1], [6, 6], [2, 2]])
C1 = K_Means(X,2)
C2 = K_Means(X,3)
C3 = K_Means_better(X2,2)
C4 = K_Means_better(X2,3)

"""
