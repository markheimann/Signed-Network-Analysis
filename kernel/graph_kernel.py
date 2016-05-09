#Graph kernels for link prediction
#Based on Kunegis et. al, 2010

#from scipy.sparse import csr_matrix, diags, pinv
import scipy.sparse as sp
from scipy.linalg import eig, cholesky, LinAlgError, pinv
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import numpy as np
import logging
import data.simulate_networks as sim
import utils.ml_pipeline as pipeline

import time
import analytics.stats as stats

#Compute signed laplacian of a matrix
#Input: matrix [sparse CSR matrix]
#Output: signed laplacian [sparse CSR matrix]
def signed_laplacian(adj_matrix):
  abs_deg_data = list()
  for row_index in range(adj_matrix.shape[0]):
    row = adj_matrix.getrow(row_index).A[0]
    #sum absolute values of all non-diagonal row elements (like Chiang et. al, 2014)
    degree = np.count_nonzero(row)
    if abs(adj_matrix[row_index,row_index]) == 1: #don't count diagonal entry
      degree -= 1 #subtract off diagonal entry if 
    abs_deg_data.append(degree)
  abs_deg_matrix = sp.diags([abs_deg_data],[0],format="csr") #make diagonal matrix with this data
  lap = abs_deg_matrix - adj_matrix
  print adj_matrix.A
  print abs_deg_matrix.A
  print lap.A
  return lap

#Get kernel matrix based on signed Laplacian
#Input: adjacency matrix [sparse CSR matrix]
#       type of kernel: string -- signed_resistance, regularized_laplacian, or heat_diffusion
#       regularization amount (for regularized laplacian and heat diffusion) [float]
#Output: kernel matrix [numpy matrix] (since it will be dense)
def sl_graph_kernel(adj_matrix, kernel_type = "signed_resistance", regularization = 0.1):
  kernel_matrix = None
  signed_lap = signed_laplacian(adj_matrix) #need this for kernels
  if kernel_type is "signed_resistance":
    print "computing pinv"
    #NOTE: numpy pinv segfaults on large matrices so use scipy instead
    #kernel_matrix = np.linalg.pinv(signed_lap.todense()) #will be dense anyway
    kernel_matrix = pinv(signed_lap.todense()) #will be dense anyway
    print kernel_matrix
  elif kernel_type is "regularized_laplacian":
    kernel_matrix = sp.identity(adj_matrix.shape[0]) + signed_lap.multiply(regularization)
    kernel_matrix = pinv(kernel_matrix.todense()) #will be dense anyway
  elif kernel_type is "heat_diffusion":
    rows, cols = adj_matrix.nonzero()
    kernel_matrix_data = list()
    for data_point_index in range(len(rows)):
      datum = adj_matrix[rows[data_point_index], cols[data_point_index]]
      kernel_matrix_data.append(np.exp(-regularization * datum))
    kernel_matrix = sp.csr_matrix((kernel_matrix_data, (rows, cols)), shape=adj_matrix.shape)
    kernel_matrix = kernel_matrix.todense() #should just be a dense matrix
  else:
    raise ValueError("unrecognized kernel type " + kernel_type)
  return kernel_matrix#.todense()

#TODO SWEng this code is duplicated in moi, hoc kfold CV split
#make pipeline more modular

#Compute k-fold cross validation using kernel methods
#Input: adjacency matrix
#       number of folds
#Output: accuracy, false positive rate, running time info
def kfoldcv(adj_matrix, num_folds = 10):
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
    #NOTE: should still be nxn so every vertex, train or test, has an entry
    #just don't use test labels when computing it?
    #kernel_type = "heat_diffusion" #still doesn't work
    #kernel_type = "regularized_laplacian" #works on simulated data but not real data?
    #kernel_type = "signed_resistance" #works on simulated data but not real data?
    reg = 0.5
    train_kernel_matrix = sl_graph_kernel(train_matrix, kernel_type, reg) 
    print train_matrix.A
    print train_kernel_matrix
    train_kernel_entries = np.asarray([train_kernel_matrix[point] for point in train_points])
    train_kernel_entries = train_kernel_entries.reshape((train_kernel_entries.size, 1))
    print train_kernel_entries.shape
    test_kernel_entries = np.asarray([train_kernel_matrix[point] for point in test_points])
    test_kernel_entries = test_kernel_entries.reshape((test_kernel_entries.size, 1))
    
    #custom kernel function based on kernel matrix
    #input: 1xn_1, 1xn2 matrix consisting of lists of vertex indices (row/col in kernel matrix)
    #output: n1 x n2 matrix consisting of kernel entries at those indices
    #def get_kernel_entries(samples1, samples2):
    #  return train_kernel_matrix[samples1,:][:,samples2]

    #train logistic regression on kernel evaluations of nodes in edge
    clf = LogisticRegression()
    clf.fit(train_kernel_entries, train_labels)

    #Make predictions
    test_preds = clf.predict(test_kernel_entries)
    #print test_preds

    #Evaluate
    acc, fpr = pipeline.evaluate(test_preds, test_labels)
    accuracy_fold_data.append(acc)
    false_positive_rate_fold_data.append(fpr)

  avg_acc = sum(accuracy_fold_data) / float(len(accuracy_fold_data))
  avg_fpr = sum(false_positive_rate_fold_data) / float(len(false_positive_rate_fold_data))
  return avg_acc, avg_fpr

if __name__ == "__main__":
  simulated = False
  real = True

  '''
  cluster_sizes = [2,3,4]
  sparsity_level = 0.5
  noise_prob = 0
  adj_matrix = sim.sample_network(cluster_sizes, sparsity_level, noise_prob)
  #print adj_matrix.A
  signed_laplacian(adj_matrix).A
  '''

  adj_matrix = None
  if simulated:
    cluster_sizes = [500,500]#[100,200,300,400]
    sparsity_level = 0.5#0.01175
    noise_prob = 0
    print "creating adjacency matrix..."
    adj_matrix = sim.sample_network(cluster_sizes, sparsity_level, noise_prob)

  elif real:
    #data_file_name = "Preprocessed Data/small_network.npy"
    data_file_name = "../data/Preprocessed Data/wiki_elections_csr.npy"
    try:
      adj_matrix = np.load(data_file_name).item()
    except Exception as e:
      raise ValueError("could not load adj matrix from file: ", e)

  avg_acc, avg_fpr = kfoldcv(adj_matrix, num_folds = 20)
  print("Accuracy %f and false positive rate %f" % (avg_acc, avg_fpr))







  