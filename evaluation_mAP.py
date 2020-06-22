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

import numpy as np

def eva(GTs, PREDs, topK):

    for i in range(np.shape(GTs)[0]):
        for j in range(np.shape(GTs)[1]):
            if GTs[i][j] > 0:
                GTs[i][j] = 1
            else:
                GTs[i][j] = 0

    hardPREDs = np.zeros(np.shape(PREDs))
    for n in range(np.shape(GTs)[1]):
        gt = np.array(GTs)[:, n]
        confidence = np.array(PREDs)[:, n]

        for j in range(np.shape(confidence)[0]):
            confidence[j] = -confidence[j]

        so = np.sort(confidence)
        si = np.argsort(confidence)

        si = si[:topK]

        hardPREDs[si, n] = 1

    retrievedInd = np.sum(hardPREDs*GTs, axis = 1)
    ep = pow(2, -52)
    precInd = retrievedInd / np.maximum( np.sum(hardPREDs, axis = 1), ep)
    prec = np.mean(precInd)

    recInd = retrievedInd / np.maximum( np.sum(GTs, axis = 1), ep)
    rec = np.mean(recInd)

    f1Ind = (2*precInd) * recInd / np.maximum((precInd + recInd), ep)
    f1 = 2*prec*rec/(prec+rec)

    
    for i in range(np.shape(retrievedInd)[0]):
        if retrievedInd[i] > 0:
            retrievedInd[i] = 1
        else:
            retrievedInd[i] = 0

    retrieved = np.sum(retrievedInd)

    m = np.shape(PREDs)[0]
    n = np.shape(PREDs)[1]
    for i in range(np.shape(GTs)[0]):
       for j in range(np.shape(GTs)[1]):
           if GTs[i][j] == -1:
               GTs[i][j] = 0

    positive_instance_vector = np.sum(GTs, axis=1)
    AP_vector = np.zeros([m, 1])

    for c in range(m):
        n_c = int(positive_instance_vector[c])
        positive_instance_location_c = np.array(np.nonzero(GTs[c]))

        a = PREDs[c][::-1]
        ranking_c_index = np.argsort(-np.array(PREDs[c]))
        ranking_c_index = np.array(ranking_c_index).reshape(1, np.shape(ranking_c_index)[0])

        precision_vector = np.zeros([n_c, 1])  ########### change  #######

        for i in range(1, int(positive_instance_vector[c] + 1)):
            precision_vector[i - 1] = max(
                np.shape((list(set(ranking_c_index[0][:i]).intersection(set(positive_instance_location_c[0])))))) / i

        AP_vector[c] = np.mean(precision_vector)
    MAP = np.mean(AP_vector)

    return prec, rec, f1, retrieved, f1Ind, precInd, recInd, AP_vector, MAP
