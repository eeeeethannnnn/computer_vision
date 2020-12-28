from typing import Tuple

import numpy as np


def compute_feature_distances(features1: np.ndarray,
                              features2: np.ndarray) -> np.ndarray:
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    dists = np.empty((features1.shape[0], features2.shape[0]))
    for i in range(features1.shape[0]):
        for j in range(features2.shape[0]):
            dists[i, j] = np.linalg.norm(features1[i, :] - features2[j, :])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features(features1: np.ndarray,
                   features2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform nearest-neighbor matching with ratio test.

    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    The results should be sorted in descending order of confidence.

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)


    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    # dist = compute_feature_distances(features1, features2)
    # matches = np.empty((0,2))
    # confidences = np.empty((0,1))
    # for i in range(dist.shape[0]):
    #     first, second = np.partition(dist[i,:], 1)[0:2]
    #     if first / second > 0.7:
    #         i, j = np.where(dist == first)
    #         matches = np.stack((np.asarray(i), np.asarray(j)), axis=-1)
    #         confidences = np.append(confidences, first)
    # confidences = np.sort(confidences, axis=None)
    dist = compute_feature_distances(features1, features2)
    dist_sort = np.sort(dist, axis=1)
    ratioTest = dist_sort[:, 0] / dist_sort[:, 1]
    tune = 0.7
    confidences = ratioTest[ratioTest < tune]
    index = np.argsort(dist, axis=1)
    tmp = np.transpose(np.nonzero(ratioTest < tune)).flatten()
    matches = np.stack([tmp, index[ratioTest < tune, 0]], axis=1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences
