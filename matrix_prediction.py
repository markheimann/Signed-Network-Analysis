#Prediction framework for global methods
#based on matrix factorization and clustering

import svp_sign_prediction as svp
import matrix_factorization as mf
import logging

import numpy as np
from scipy.sparse import csr_matrix
import ml_pipeline as pipeline

#k fold cross validation for matrix completion problems
# Input: adjacency matrix
# Algorithm to use
# Parameters for that algorithm (loss function, learning rate, etc.)
# (for more details see individual algorithm requirements in matrix_factorization.py)
# Number of folds for cross-valudation
def kfold_CV_pipeline(adj_matrix, alg, alg_params, num_folds=10):
  #get folds
  nonzero_row_indices, nonzero_col_indices = adj_matrix.nonzero()
  data = zip(nonzero_row_indices, nonzero_col_indices) #TODO maybe should try to keep arrays separate?
  labels = adj_matrix[nonzero_row_indices, nonzero_col_indices]
  folds = pipeline.kfold_CV_split(data, num_folds)
  print "got folds"

  #keep track of accuracy, false positive rate
  accuracy = 0
  false_positive_rate = 0

  #perform learning problem on each fold
  for fold_index in range(num_folds):
    print("Fold %d" % (fold_index + 1))
    #get train data for learning problem
    train_points = pipeline.join_folds(folds, fold_index)
    train_row_indices, train_col_indices = zip(*train_points)
    train_labels = adj_matrix[train_row_indices, train_col_indices].A[0] #array of signs of training edges
    #construct matrix using just training edges
    train_matrix = csr_matrix((train_labels, (train_row_indices, train_col_indices)), shape = adj_matrix.shape)

    #get test data
    test_points = folds[fold_index]
    test_row_indices, test_col_indices = zip(*test_points)
    test_labels = adj_matrix[test_row_indices, test_col_indices].A[0] #array of signs of test edges

    #perform learning on training matrix
    train_complet = None #completed matrix to use for prediction

    alg = alg.lower()
    if alg == "svp":
      try:
        rank,tol,max_iter,step_size = alg_params
        train_complet = svp.sign_prediction_SVP(train_matrix, rank, tol,
                                      max_iter, step_size)
      except Exception: 
        logging.exception("Exception: ")
        raise ValueError("Could not perform SVP. Check input?")
    elif alg == "sgd":
      try:
        learn_rate, loss_type, tol, max_iter, reg_param, dim = alg_params
        factor1, factor2 = mf.matrix_factor_SGD(train_matrix, learn_rate, 
                                          loss_type, tol, max_iter, 
                                          reg_param, dim)
        train_complet = csr_matrix.sign(csr_matrix(factor1*factor2.transpose()))
      except Exception:
        logging.exception("Exception: ")
        e.printStackTrace()
        raise ValueError("Could not perform SGD. Check input?")
    elif alg == "als":
      try:
        max_iter, dim = alg_params
        factor1, factor2 = mf.matrix_factor_ALS(train_matrix, dim, max_iter)
        train_complet = csr_matrix.sign(csr_matrix(factor1.transpose()*factor2))
      except Exception as e:
        logging.exception("Exception: ")
        raise ValueError("Could not perform ALS. Check input?")

    #WRITETEST to make sure this is same shape as adj matrix
    print train_complet.shape

    preds = train_complet[test_row_indices, test_col_indices]

    acc, fpr = pipeline.evaluate(preds, test_labels)
    accuracy += acc
    false_positive_rate += fpr

  #Return average accuracy and false positive rate
  accuracy = accuracy / num_folds
  false_positive_rate = false_positive_rate/num_folds
  return accuracy, false_positive_rate

if __name__ == "__main__":
  data_file_name = "Preprocessed Data/wiki_elections_csr.npy"
  #EXCEPTIONHANDLING makes sure this file exists
  adj_matrix = np.load(data_file_name).item()

  NUM_VERTICES = 100 #take a small part of dataset
  adj_matrix = csr_matrix(adj_matrix.todense()[:NUM_VERTICES,:NUM_VERTICES])
  num_folds = 10

  use_svp = True
  use_sgd = False
  use_als = False

  alg = ""
  alg_params = None

  #settings if using SVP
  if use_svp:
    #Parameters used for this experiment
    rank = 5
    tol = 1
    max_iter = 10
    step_size = 1

    #Bundle up these parameters and use this algorithm
    alg_params = (rank, tol, max_iter, step_size)
    alg = "svp"

  #settings if using SGD
  elif use_sgd:
    #Parameters used for this experiment
    learning_rate = 1
    loss_type = "sigmoid"
    tol = 1
    max_iter = 5
    reg_param = 1
    dim = 20

    #Bundle up these parameters and use this algorithm
    alg_params = (learning_rate, loss_type, tol, max_iter, reg_param, dim)
    alg = "sgd"

  #settings if using als
  elif use_als:
    #Parameters used for this experiment
    max_iter = 5
    dim = 20

    #Bundle up these parameters and use this algorithm
    alg_params = (max_iter, dim)
    alg = "als"

  avg_accuracy, avg_false_positive_rate = kfold_CV_pipeline(adj_matrix, alg, alg_params, num_folds)
  print "Average accuracy: ", avg_accuracy
  print "Average false positive rate: ", avg_false_positive_rate