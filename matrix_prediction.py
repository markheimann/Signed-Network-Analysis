#Prediction framework for global methods
#based on matrix factorization and clustering

import svp_sign_prediction as svp
import matrix_factorization as mf

import numpy as np
from scipy.sparse import csr_matrix
import hoc_learning as hoc #to get k-fold CV 
#TODO rethink software eng to make hoc-learning more generic

#k fold cross validation for matrix completion problems
def kfold_CV_pipeline(adj_matrix, num_folds=10):
  #get folds
  nonzero_row_indices, nonzero_col_indices = adj_matrix.nonzero()
  data = zip(nonzero_row_indices, nonzero_col_indices) #TODO maybe should try to keep arrays separate?
  labels = adj_matrix[nonzero_row_indices, nonzero_col_indices]
  folds = hoc.kfold_CV_split(data, num_folds)
  print "got folds"

  #keep track of accuracy, false positive rate
  accuracy = 0
  false_positive_rate = 0

  #perform learning problem on each fold
  for fold_index in range(num_folds):
    print("Fold %d" % (fold_index + 1))
    #get train data for learning problem
    train_points = hoc.join_folds(folds, fold_index)
    train_row_indices, train_col_indices = zip(*train_points)
    train_labels = adj_matrix[train_row_indices, train_col_indices].A[0] #array of signs of training edges
    #construct matrix using just training edges
    train_matrix = csr_matrix((train_labels, (train_row_indices, train_col_indices)), shape = adj_matrix.shape)
    print "created train matrix"

    #get test data
    test_points = folds[fold_index]
    test_row_indices, test_col_indices = zip(*test_points)
    test_labels = adj_matrix[test_row_indices, test_col_indices].A[0] #array of signs of test edges
    print "created test matrix"

    #perform learning on training matrix
    '''
    rank = 5
    tol = 1
    max_iter = 10
    step_size = 1
    train_complet = svp.sign_prediction_SVP(train_matrix, rank, tol,
                                      max_iter, step_size)
    '''
    learning_rate = 1
    loss_type = "sigmoid"
    tol = 1
    max_iters = 5
    regularization_param = 1
    dim = 20

    factor1, factor2 = mf.matrix_factor_SGD(train_matrix, learning_rate, 
                                          loss_type, tol, max_iters, 
                                          regularization_param, dim)
    print factor1.shape, factor2.shape
    print factor1
    train_complet = csr_matrix.sign(csr_matrix(factor1*factor2.transpose() ))
    print train_complet

    preds = train_complet[test_row_indices, test_col_indices]

    #again consider how HOC can be more general
    acc, fpr = hoc.evaluate(preds, test_labels)
    accuracy += acc
    false_positive_rate += fpr

  #Return average accuracy and false positive rate
  accuracy = accuracy / num_folds
  false_positive_rate = false_positive_rate/num_folds
  return accuracy, false_positive_rate

if __name__ == "__main__":
  data_file_name = "Preprocessed Data/wiki_elections_csr.npy"
  adj_matrix = np.load(data_file_name).item()
  NUM_VERTICES = 100 #take a small part of dataset
  adj_matrix = csr_matrix(adj_matrix.todense()[:NUM_VERTICES,:NUM_VERTICES])
  num_folds = 10
  avg_accuracy, avg_false_positive_rate = kfold_CV_pipeline(adj_matrix, num_folds)
  print "Average accuracy: ", avg_accuracy
  print "Average false positive rate: ", avg_false_positive_rate