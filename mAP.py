# =====================
# Dual Relation Semi-supervised Multi-label Learning
# =====================
# Author: Lichen Wang, Yunyu Liu
# Date: May, 2020
# E-mail: wanglichenxj@gmail.com, liu3154@purdue.edu

# @inproceedings{DRML_AAAI20,
#   title={Dual Relation Semi-supervised Multi-label Learning},
#   author={Wang, Lichen and Liu, Yunyu and Qin, Can and Sun, Gan and Fu, Yun},
#   booktitle={Proceedings of AAAI Conference on Artificial Intelligence},
#   year={2020}
# }
# =====================

def AP_information_retrieval(label_matrix_prediction, label_matrix_gt):
    import numpy as np
    m = np.shape(label_matrix_prediction)[0]
    n = np.shape(label_matrix_prediction)[1]
    for i in range(np.shape(label_matrix_gt)[0]):
        for j in range(np.shape(label_matrix_gt)[1]):
            if label_matrix_gt[i][j] == -1:
                label_matrix_gt[i][j] = 0

    positive_instance_vector = np.sum(label_matrix_gt, axis = 1)
    AP_vector = np.zeros([m, 1])

    for c in range(m):
        n_c = positive_instance_vector[c]
        positive_instance_location_c = np.array(np.nonzero(label_matrix_gt[c]))
        a = label_matrix_prediction[c][::-1]
        ranking_c_index = np.argsort( -np.array(label_matrix_prediction[c]) )
        ranking_c_index = np.array(ranking_c_index).reshape(1, np.shape(ranking_c_index)[0])
        precision_vector = np.zeros([n_c, 1])
        for i in range(1, int(positive_instance_vector[c]+1)):
            precision_vector[i-1] = max(np.shape((list(set(ranking_c_index[0][:i]).intersection(set(positive_instance_location_c[0])))))) / i

        AP_vector[c] = np.mean(precision_vector)
    MAP = np.mean(AP_vector)

    return AP_vector, MAP