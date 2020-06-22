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

import tensorflow as tf
import numpy as np
import os
import scipy.io # Load and write MATLAB mat file
import random # Sample the label

import evaluation_mAP
import evaluation
import mAP

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Assign GPU number
print ('data loading ...')

# Load datasets
mat = scipy.io.loadmat('Dataset/cub_data_VGG_ori.mat')

Xl=mat['Xl'] # Load labeled samples
Yl=mat['Sl']

Xt=mat['Xu'] # Load unlabeled samples
Yt=mat['Su']

d = 1 # Threshold of label difference
feature_dim=len(Xl) # Featrue dimension
label_dim=len(Yl) # Label dimension
data_number_labeled=len(Xl[0]) # Number of labeled samples
data_number_test=len(Xt[0]) # Number of unlabeled samples


# Transpose data matrix
Xl=Xl.T
Xt=Xt.T
_Xt = mat['Xu'].T
Yl=Yl.T
Yt=Yt.T


# Network parameters
mb_size=64 # Batch size
h_dim=800 # Hidden layer dimension
R_h_dim = 1024 # Hidden layer dimension
R_dim = 1024 # Representation dimension
hh_dim = 1000 # Hidden layer dimension


# Define random matrix for net weights initialization
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# Define leaky-Relu activation
def leak_relu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


# Placeholders for labeled and unlabeled sample matrix and label matrix
X = tf.placeholder(tf.float32, shape=[None, feature_dim]) # Input feature data
Xu = tf.placeholder(tf.float32, shape=[None, feature_dim]) # Input feature data
y = tf.placeholder(tf.float32, shape=[None, label_dim]) # Input label data


# ============================
#        Classifier-1
# ============================
# Layer-1 weights
C1_W1 = tf.Variable(xavier_init([R_dim, h_dim]))
C1_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
# Layer-2 weights
C1_W2 = tf.Variable(xavier_init([h_dim, label_dim]))
C1_b2 = tf.Variable(tf.zeros(shape=[label_dim]))

theta_C1 = [C1_W1, C1_W2, C1_b1, C1_b2]

# Define classifier-1
def classifier_1(X):
    inputs = X
    C_h1 = tf.nn.relu(tf.matmul(inputs, C1_W1) + C1_b1)
    C_log_prob = tf.matmul(C_h1, C1_W2) + C1_b2
    C_prob = tf.nn.sigmoid(C_log_prob)
    return C_prob # , C_feature


# ============================
#        Classifier-2
# ============================
# Layer-1 weights
C2_W1 = tf.Variable(xavier_init([R_dim, h_dim]))
C2_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
# Layer-2 weights
C2_W2 = tf.Variable(xavier_init([h_dim, label_dim]))
C2_b2 = tf.Variable(tf.zeros(shape=[label_dim]))

theta_C2 = [C2_W1, C2_W2, C2_b1, C2_b2]

# Define classifier-2
def classifier_2(X):
    inputs = X
    C_h1 = tf.nn.relu(tf.matmul(inputs, C2_W1) + C2_b1)
    C_log_prob = tf.matmul(C_h1, C2_W2) + C2_b2
    C_prob = tf.nn.sigmoid(C_log_prob)
    return C_prob # , C_feature


# ==================
# Subspace projector
# ==================
# Layer-1 weights
R1_W1 = tf.Variable(xavier_init([feature_dim, R_h_dim]))
R1_b1 = tf.Variable(tf.zeros(shape=[R_h_dim]))
# Layer-2 weights
R1_W2 = tf.Variable(xavier_init([R_h_dim, R_dim]))
R1_b2 = tf.Variable(tf.zeros(shape=[R_dim]))

theta_R1 = [R1_W1, R1_b1, R1_W2, R1_b2]

def Represent_1(X):
    inputs = X
    R_log_prob1 = leak_relu(tf.matmul(inputs, R1_W1) + R1_b1, 0.1)
    R_log_prob2 = tf.matmul(R_log_prob1, R1_W2) + R1_b2    
    return R_log_prob2


# ======================
# Label-relation network
# ======================
# Layer-1 weights
Graph_W1 = tf.Variable(xavier_init([label_dim*label_dim, hh_dim]))
Graph_b1 = tf.Variable(tf.zeros(shape=[hh_dim]))
# Layer-2 weights
Graph_W2 = tf.Variable(xavier_init([hh_dim, label_dim]))
Graph_b2 = tf.Variable(tf.zeros(shape=[label_dim]))

theta_Graph = [Graph_W1, Graph_W2, Graph_b1, Graph_b2]

# Define Label-relation network
def graph(C_prob1, C_prob2):
    # Initial prediction of C1 and C2
    C_prob_1 = tf.expand_dims(C_prob1, -1)
    C_prob_2 = tf.expand_dims(C_prob2, 1)
    # Get Label correlation graph
    W_feature = tf.matmul(C_prob_1, C_prob_2) 
    W_feature_1 = tf.reshape(W_feature, [-1, label_dim*label_dim])
    C_h2 = tf.nn.relu(tf.matmul(W_feature_1, Graph_W1) + Graph_b1)
    C_feature = tf.nn.sigmoid(tf.matmul(C_h2, Graph_W2) + Graph_b2)
    return C_feature


# ====================
# Define loss function
# ====================
X_r = Represent_1(X) # Labeled samples
classified_label_X_prob_1 = classifier_1(X_r)
classified_label_X_prob_2 = classifier_2(X_r)
classified_label_X_graph = graph(classified_label_X_prob_1, classified_label_X_prob_2)

Xu_r = Represent_1(Xu) # Unlabeled samples
classified_unlabel_X_prob_1 = classifier_1(Xu_r)
classified_unlabel_X_prob_2 = classifier_2(Xu_r)

C_loss_sum_graph = tf.reduce_mean(tf.square(y-classified_label_X_graph))
C_loss_1 = tf.reduce_mean(tf.square(y-classified_label_X_prob_1))
C_loss_2 = tf.reduce_mean(tf.square(y-classified_label_X_prob_2))

DA_loss = tf.reduce_mean(tf.norm(classified_unlabel_X_prob_1 - classified_unlabel_X_prob_2, ord=1)) * 0.00001

# The sum of the classification loss
C_loss = C_loss_1 + C_loss_2 + 0.2 * C_loss_sum_graph

# Classification loss for updating P, C1 and C2
stepA = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(C_loss, var_list=[theta_C1, theta_C2, theta_R1])
# Adversarial similarity loss for updating C1 and C2
stepB = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(C_loss * 1.0 -DA_loss, var_list=[theta_C1, theta_C2])
stepC = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(DA_loss, var_list=[theta_R1])
# Classification loss for updating C_R
G_solver_sum = tf.train.AdamOptimizer(learning_rate=0.001).minimize(C_loss, var_list=[theta_Graph])

# The co-training confidence and consistence
dis_C1_C2 = tf.reduce_sum(tf.square(classified_label_X_prob_1 - classified_label_X_prob_2),1)
dis_C1_G = tf.reduce_sum(tf.square(classified_label_X_prob_1 - classified_label_X_graph), 1)
dis_C2_G = tf.reduce_sum(tf.square(classified_label_X_graph - classified_label_X_prob_2), 1)


# ==============
# Model training
# ==============
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

for seperate_train in range(150):
    
    # =====================
    # Training all networks
    # =====================
    for it in range(100):
        # Get training batch
        rand_idx = random.sample(range(data_number_labeled), mb_size)
        X_mb = Xl[rand_idx,:]
        y_mb = Yl[rand_idx,:]
        
        # StepA        
        for _ in range(1):            
            _, C_loss_1_curr, C_loss_2_curr = sess.run([stepA, C_loss_1, C_loss_2], feed_dict={X: X_mb, y: y_mb})

        mb_size_unlabel = min(mb_size, data_number_test)
        rand_idx = random.sample(range(data_number_test),mb_size_unlabel)
        Xu_mb = Xt[rand_idx,:]
        
        # StepB        
        for _ in range(1):
            _, DA_curr = sess.run([stepB, DA_loss], feed_dict={X:X_mb, y:y_mb, Xu:Xu_mb})
        
        # StepC        
        for _ in range(1):
            _, DA_curr = sess.run([stepC, DA_loss], feed_dict={Xu:Xu_mb})
        for _ in range(1):
            _, graph_loss_curr = sess.run([G_solver_sum, C_loss_sum_graph], feed_dict={X:X_mb, y:y_mb})
    
    
    # =======================
    # Pscudo label assignment
    # =======================
    if seperate_train>50:
        for _ in range(1):            
            if data_number_test <= 0:
                break            
            batch_num = data_number_test       
            result_graph, dis_C1_C2_res, dis_C1_G_res, dis_C2_G_res = sess.run([classified_label_X_graph, dis_C1_C2, dis_C1_G, dis_C2_G], feed_dict={X:Xt})
            f = [0 for _ in range(batch_num)]
            X_list_sure = []
            Y_list_sure = []
            X_list_unsure = []
            for i in range(batch_num):
                if dis_C1_C2_res[i] < d:                
                    X_list_sure.append(Xt[i])
                    Y_list_sure.append(result_graph[i])
                    data_number_test -= 1
                    data_number_labeled += 1
                else:
                    X_list_unsure.append(Xt[i])

            if data_number_test == batch_num:
                continue
            Xt = np.array(X_list_unsure)            
            Xl = np.append(Xl, np.array(X_list_sure), axis=0)
            Yl = np.append(Yl, np.array(Y_list_sure), axis=0)

    # Evaluate classification performance
    if seperate_train % 1 == 0:
        # Prediction labels
        cls_label_Ft_1, cls_label_Ft_2, cls_label_Ft_3 = sess.run([classified_label_X_prob_1, classified_label_X_prob_2, classified_label_X_graph], feed_dict={X:_Xt})        

        # Show performance
        print('====================')        
        print("Iter =  ", seperate_train)
        # Performance of C1
        prec, rec, f1, retrieved, f1Ind, precInd, recInd = evaluation.eva(Yt.T, cls_label_Ft_1.T, 5)
        print('C1: Pre=%.4f ' %prec, 'Rec=%.4f ' %rec, 'F1=%.4f ' %f1, 'N-R=%d ' %retrieved)
        # Performance of C2
        prec, rec, f1, retrieved, f1Ind, precInd, recInd = evaluation.eva(Yt.T, cls_label_Ft_2.T, 5)
        print('C2: Pre=%.4f ' %prec, 'Rec=%.4f ' %rec, 'F1=%.4f ' %f1, 'N-R=%d ' %retrieved)
        # Final performance
        prec, rec, f1, retrieved, f1Ind, precInd, recInd, _, mAP = evaluation_mAP.eva(Yt.T, cls_label_Ft_3.T, 5)
        print('Final: Pre=%.4f ' %prec, 'Rec=%.4f ' %rec, 'F1=%.4f ' %f1, 'N-R=%d ' %retrieved, 'mAP=%.4f ' %mAP)

