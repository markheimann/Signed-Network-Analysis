#Use local method (HOC) for sign prediction in signed networks
#Based on Chiang et. al, 2014

import numpy as np
import cPickle
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import norm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
import hoc_edge_features as hoc
import ml_pipeline as pipeline
import random, os
import simulate_networks as sim

#Perform cross validation, testing on one fold and training on the rest
#Input: adjacency matrix
#       indices of data points in each folds
#       Features, labels to learn from
#Output: average test accuracy, false positive rate
def kfold_CV(adj_matrix, folds, num_features = -1):
  num_folds = len(folds)
  accuracy = 0
  false_positive_rate = 0
  for fold_index in range(num_folds):
    print("Fold %d:" % (fold_index + 1))

    #get data
    train_points = pipeline.join_folds(folds, fold_index)
    test_points = folds[fold_index]   
    train_test_overlap = False

    train_row_indices, train_col_indices = zip(*train_points)
    test_row_indices, test_col_indices = zip(*test_points)
    train_labels = adj_matrix[train_row_indices, train_col_indices].A[0] #array of signs of training edges

    #construct matrix using just training edges
    train_matrix = csr_matrix((train_labels, (train_row_indices, train_col_indices)), shape = adj_matrix.shape)
    train_matrix = (train_matrix + train_matrix.transpose()).sign() #make symmetric
    feature_products = hoc.extract_edge_features(train_matrix, "whatever", max_cycle_order = 4)

    #get features and labels corresponding to each data point
    train_data = np.asarray([hoc.extract_features_for_edge(feature_products, tr_point) for tr_point in train_points])
    train_labels = adj_matrix[train_row_indices, train_col_indices].A[0] #array of signs of training edges
    test_data = np.asarray([hoc.extract_features_for_edge(feature_products, te_point) for te_point in test_points])
    test_labels = adj_matrix[test_row_indices, test_col_indices].A[0] #array of signs of test edges

    if num_features > 0:
      feat_sel = SelectKBest(f_classif, k=num_features)
      feat_sel.fit(train_data, train_labels)
      train_data = feat_sel.transform(train_data)
      test_data = feat_sel.transform(test_data)

    elif num_features == 0: #train on random features
      print "train data: random matrix of shape ", train_data.shape
      train_data = np.random.random(train_data.shape)
    
    print "number of features: ", train_data.shape[1]

    #train logistic regression classifier
    clf = LogisticRegression()
    clf.fit(train_data, train_labels)

    #Evaluate
    test_preds = clf.predict(test_data)

    #average prediction/label tells you what min and max are 
    #(if it's strictly between -1 and 1 there are both positive and negatives)
    acc, fpr = pipeline.evaluate(test_preds, test_labels)
    accuracy += acc
    false_positive_rate += fpr

  accuracy = accuracy / num_folds
  false_positive_rate = false_positive_rate/num_folds
  return accuracy, false_positive_rate

#Machine learning pipeline for prediction using HOC features
#Feature extraction to model training and usage
#Input: adjacency matrix (data)
#       Name of dataset to use
#       Maximum cycle order to consider
#       Number of folds for k-fold cross validation (default 10 like in the paper)
#       Number of features to use (to test whether classifier is actually learning)
#Output: average accuracy, false positive rate across folds
def hoc_learning_pipeline(adj_matrix, dataset_name, max_cycle_order, num_folds=10, num_features=-1):
  #Get data
  features_dict, labels_dict = ({},{})#hoc.extract_edge_features(adj_matrix, dataset_name, max_cycle_order, dataset_name)
  #print "number of features calculated: ", len(features_dict[features_dict.keys()[0]])

  #for key in features_dict.keys():
  #  features_dict[key] = features_dict[key][11:12]

  #completely randomize the features
  #NOTE: with this test, classifier should just predict mode label
  if num_features == 0:
    for key in features_dict.keys():
      features_dict[key] = list(np.random.random(len(features_dict[key])))

  #choose only a subset of the features to learn from
  #note: fewer features (e.g. 4) --> classifier always predicts mode label

  #Split into folds
  unique_edge_list = get_unique_edges(adj_matrix)
  data_folds = pipeline.kfold_CV_split(unique_edge_list, num_folds)

  #Perform k-fold cross validation
  avg_accuracy, avg_false_positive_rate = kfold_CV(adj_matrix, data_folds, num_features)
  return avg_accuracy, avg_false_positive_rate

#get unique edges in adjacency matrix
def get_unique_edges(adj_matrix):
  rows,cols = adj_matrix.nonzero()
  unique_edges = set()
  for edge_index in range(len(rows)):
    edge = (rows[edge_index],cols[edge_index])
    if edge not in unique_edges and edge[::-1] not in unique_edges:
      unique_edges.add(edge)
  unique_edge_list = list(unique_edges)
  return unique_edge_list

if __name__ == "__main__":
  #data_file_name = "Preprocessed Data/wiki_elections_csr.npy"
  #dataset_name = "wikipedia"
  #data_file_name = "Preprocessed Data/Slashdot090221_csr.npy"
  #dataset_name = "slashdot"
  #data_file_name = "Preprocessed Data/epinions_csr.npy"
  #dataset_name = "epinions"
  #data_file_name = "Preprocessed Data/small_network.npy"
  #dataset_name = "small"
  dataset_name = "sim1234500"
  #if not os.path.exists(data_file_name):
  #  raise ValueError("invalid path for data file")

  adj_matrix = sim.sample_network([100,200,300,400,500],0.1,0.01)
  #adj_matrix = np.load(data_file_name).item()
  print("using %s dataset" % dataset_name)
  max_cycle_order = 3

  num_folds = 10
  num_features = -1
  avg_accuracy, avg_false_positive_rate = hoc_learning_pipeline(adj_matrix, dataset_name, max_cycle_order, num_folds, num_features)
  print "Average accuracy: ", avg_accuracy
  print "Average false positive rate: ", avg_false_positive_rate

