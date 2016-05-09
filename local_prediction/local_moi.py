#Contains methods for sign prediction in signed networks
#Local methods: method of influence (balancing cycles)
#Based on Chiang et. al, 2014

import numpy as np
import scipy.sparse as sp
from scipy.linalg import norm
import cPickle, os, time
import utils.ml_pipeline as pipeline
import analytics.stats as stats


#MOI (measures of imbalance) for link sign prediction
#Predict sign by balancing cycles up to given length
#MOI-\infty is the signed Katz measure
#Input: adjacency matrix
#       dataset name 
#       edge to get sign for
#       max cycle order to consider (np.inf for signed Katz)
#       discount factor (or single discount factor if inf) //list if finite max cycle order, otherwise single value
#Output: sign for that edge
def predict_sign_MOI(prediction_matrices, discount_factor, edge, max_cycle_order):
  prediction = None
  if max_cycle_order < 3: #cycle must have length at least 3
    raise ValueError("maximum cycle order must be at least 3")
  if max_cycle_order == np.inf: #compute signed Katz measure
    prediction = 2*(prediction_matrices[edge] >= 0) - 1
  else: #compute using formula in Lemma 11 from paper
    #compute imbalance
    imbalance = 0

    #consider up to maximum cycle order
    for cycle_order in range(3,max_cycle_order + 1):
      #subtract 3 for indexing since starting from 3
      imbalance += discount_factor[cycle_order - 3] * prediction_matrices[cycle_order - 3][edge]

    #predict and return sign: 1 if imbalance is positive, -1 otherwise
    prediction = 2*(imbalance >= 0) - 1
  return prediction

#Input: adjacency matrix
#       discount factor //list if finite max cycle order, otherwise single value
#       max cycle order to consider (np.inf for signed Katz)
#       discount factor 
#Output: matrices used to compute MOI
def get_prediction_matrices(adj_matrix, discount_factor, max_cycle_order):
  if max_cycle_order == np.inf: #compute signed Katz measure
    if type(discount_factor) is not float:
      raise ValueError("discount factor must be float") #TODO must be sufficiently small (< ||A||_2) too? 
    prediction_matrix = sp.identity(adj_matrix.shape[0])
    prediction_matrix = prediction_matrix - discount_factor * adj_matrix
    prediction_matrix = sp.linalg.inv(prediction_matrix)
    prediction_matrix = prediction_matrix - sp.identity(adj_matrix.shape[0])
    prediction_matrix = prediction_matrix - discount_factor * adj_matrix
    return prediction_matrix
  else:
    if type(discount_factor) is not list:
      raise ValueError("for finite max cycle order, must provide list of discount factors for each cycle order between 3 and max")

    #Load in products of adjacency matrix of power up to max cycle order
    #Compute if needed
    products = list()
    #'''
    current_product = sp.csr_matrix(adj_matrix)
    order = 3
    while order <= max_cycle_order:
      highest_power_product = None
      current_product = current_product.dot(adj_matrix) #compute next higher power product
      products.append(current_product) #add this to our list of products used to compute MOI
      order += 1
    return products

#Compute k-fold cross validation using MOI
#Input: adjacency matrix
#       discount factor //list if finite max cycle order, otherwise single value
#       max cycle order to consider (np.inf for signed Katz)
#       number of folds
#Output: accuracy, false positive rate, running time info
def kfoldcv_moi(adj_matrix, discount, max_cycle_order, num_folds = 10):
  unique_edge_list = pipeline.get_unique_edges(adj_matrix)
  data_folds = pipeline.kfold_CV_split(unique_edge_list, num_folds)

  accuracy_fold_data = list()
  false_positive_rate_fold_data = list()
  time_fold_data = list()
  for fold_index in range(num_folds):
    print("Fold %d:" % (fold_index + 1))

    #get data
    train_points = pipeline.join_folds(data_folds, fold_index)
    test_points = data_folds[fold_index]   
    train_test_overlap = False

    train_row_indices, train_col_indices = zip(*train_points)
    test_row_indices, test_col_indices = zip(*test_points)
    train_labels = adj_matrix[train_row_indices, train_col_indices].A[0] #array of signs of training edges
    test_labels = adj_matrix[test_row_indices, test_col_indices].A[0] #array of signs of test edges

    #construct matrix using just training edges
    train_matrix = sp.csr_matrix((train_labels, (train_row_indices, train_col_indices)), shape = adj_matrix.shape)
    train_matrix = (train_matrix + train_matrix.transpose()).sign() #make symmetric

    #Make predictions
    preds = list()
    before_train = time.time()
    prediction_matrices = get_prediction_matrices(train_matrix, discount, max_cycle_order)
    for test_point in test_points:
      predicted_sign = predict_sign_MOI(prediction_matrices, discount, test_point, max_cycle_order)
      preds.append(predicted_sign)
    after_train = time.time()
    model_time = after_train - before_train
    print "MOI model time for one fold: ", model_time
    test_preds = np.asarray(preds)

    #Evaluate
    acc, fpr = pipeline.evaluate(test_preds, test_labels)
    accuracy_fold_data.append(acc)
    false_positive_rate_fold_data.append(fpr)
    time_fold_data.append(model_time)

  avg_acc = sum(accuracy_fold_data) / float(len(accuracy_fold_data))
  avg_fpr = sum(false_positive_rate_fold_data) / float(len(false_positive_rate_fold_data))
  avg_time = sum(time_fold_data) / float(len(time_fold_data))
  acc_stderr = stats.error_width(stats.sample_std(accuracy_fold_data), num_folds)
  fpr_stderr = stats.error_width(stats.sample_std(false_positive_rate_fold_data), num_folds)
  time_stderr = stats.error_width(stats.sample_std(time_fold_data), num_folds)
  return avg_acc, acc_stderr, avg_fpr, fpr_stderr, avg_time, time_stderr
