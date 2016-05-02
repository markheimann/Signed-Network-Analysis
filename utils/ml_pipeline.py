#Contains machine learning pipeline methods
#Based on Chiang et. al, 2014

import numpy as np
import cPickle
from scipy.linalg import norm
from sklearn.linear_model import LogisticRegression
import random

#Split data into folds for cross-validation
#Input: edges in dataset [list of 2-tuples]
#       Number of folds (k in k-fold cross validation) [int]
#Output: k disjoint sets of vertices whose 
#union is the set of all edges (nonzero entries in matrix) [list of lists]
def kfold_CV_split(data_points, num_folds=10):
  random.shuffle(data_points) #shuffle data points into random order
  fold_size = len(data_points)/num_folds
  folds = list() #data points (edges) in each fold
  for fold_index in range(num_folds): #append evenly sized folds of data
    if fold_index == num_folds - 1: #last fold--append remaining data
      folds.append(data_points[fold_size*fold_index:])
    else:
      folds.append(data_points[fold_size*fold_index:fold_size*(fold_index + 1)])
  return folds

#Join folds other than the one use for testing to construct training dataset
#Input: all folds [list of lists]
# Index of fold to leave out [int]
#Output: List of training points (all other folds) [list]
def join_folds(folds, fold_to_leave_out):
  initial_fold_number = 0
  if fold_to_leave_out == 0:
    initial_fold_number = 1
  #initialize fold data and labels
  data = list(folds[initial_fold_number]) #copy by value
  for fold_index in range(1,len(folds)):
    if fold_to_leave_out == 0 and fold_index == 1:
      continue #don't re-add first fold
    if fold_index != fold_to_leave_out:
      #add data
      fold_data = folds[fold_index]
      data += fold_data
  return data

#Given test predictions and labels, evaluate metrics like accuracy
#Input: predictions [np array]
#       labels [np array]
#Output: accuracy [float 0-1]
#       false positive rate [float 0-1]
#Action: print diagnostics too
def evaluate(predictions, labels):
  #average prediction tells you if mostly one label predicted
  print("Predictions: avg %f" % np.mean(predictions))
  print("Labels: avg %f" % np.mean(labels))
  accuracy = np.mean(predictions == labels)
  print("Accuracy: %f" % accuracy)
  
  #false positives: prediction 1 but actual label -1
  num_false_positives = np.sum(predictions == labels + 2)
  #test predictions and labels both -1
  num_true_negatives = np.sum(np.logical_and(predictions == -1,labels == -1))
  false_positive_rate = 0
  try:
    false_positive_rate = float(num_false_positives) / (num_false_positives + num_true_negatives)
  except ZeroDivisionError:
    print "OK...so no false positives and no true negatives? hmmm..."
  print("False positive rate: %f" % false_positive_rate)
  return accuracy, false_positive_rate

#get unique edges in adjacency matrix
#Input: adjacency matrix [sparse csr matrix]
#Output: list of unique edges [list of 2-tuples of ints]
def get_unique_edges(adj_matrix):
  rows,cols = adj_matrix.nonzero()
  unique_edges = set()
  for edge_index in range(len(rows)):
    edge = (rows[edge_index],cols[edge_index])
    if edge not in unique_edges and edge[::-1] not in unique_edges:
      unique_edges.add(edge)
  unique_edge_list = list(unique_edges)
  return unique_edge_list
