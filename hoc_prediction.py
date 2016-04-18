#Use local method (HOC) for sign prediction in signed networks
#Based on Chiang et. al, 2014

import numpy as np
import cPickle, time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import norm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
import hoc_edge_features as hoc
import ml_pipeline as pipeline
import random, os
import simulate_networks as sim
import analytics

#Perform cross validation, testing on one fold and training on the rest
#Input: adjacency matrix
#       indices of data points in each folds
#       Features, labels to learn from
#Output: average test accuracy, false positive rate
def kfold_CV(adj_matrix, folds, max_cycle_order, num_features = -1):
  num_folds = len(folds)
  accuracy_fold_data = list()
  false_positive_rate_fold_data = list()
  time_fold_data = list()
  for fold_index in range(num_folds):
    print("Fold %d:" % (fold_index + 1))

    #get data
    train_points = pipeline.join_folds(folds, fold_index)
    test_points = folds[fold_index]   
    train_test_overlap = False

    train_row_indices, train_col_indices = zip(*train_points)
    test_row_indices, test_col_indices = zip(*test_points)
    train_labels = adj_matrix[train_row_indices, train_col_indices].A[0] #array of signs of training edges
    test_labels = adj_matrix[test_row_indices, test_col_indices].A[0] #array of signs of test edges

    #construct matrix using just training edges
    train_matrix = csr_matrix((train_labels, (train_row_indices, train_col_indices)), shape = adj_matrix.shape)
    train_matrix = (train_matrix + train_matrix.transpose()).sign() #make symmetric

    #Compute feature products
    #This dominates the training time, so report time for only this part for experiments
    before_train = time.time()
    feature_products = hoc.extract_edge_features(train_matrix, max_cycle_order)

    #get features and labels corresponding to each data point
    train_data = np.asarray([hoc.extract_features_for_edge(feature_products, tr_point) for tr_point in train_points])
    test_data = np.asarray([hoc.extract_features_for_edge(feature_products, te_point) for te_point in test_points])
    after_train = time.time()
    model_time = after_train - before_train

    #if, for experimental reasons, we don't want to train on all the features instead
    #as a diagnostic for what the model is actually learning and why
    if num_features > 0: #perform feature selection
      feat_sel = SelectKBest(f_classif, k=num_features)
      feat_sel.fit(train_data, train_labels)
      train_data = feat_sel.transform(train_data)
      test_data = feat_sel.transform(test_data)
    elif num_features == 0: #train on random features
      print "train data: random matrix of shape ", train_data.shape
      train_data = np.random.random(train_data.shape)
    
    #print "number of features: ", train_data.shape[1]

    #train logistic regression classifier
    clf = LogisticRegression()
    clf.fit(train_data, train_labels)

    #Evaluate
    test_preds = clf.predict(test_data)

    acc, fpr = pipeline.evaluate(test_preds, test_labels)
    accuracy_fold_data.append(acc)
    false_positive_rate_fold_data.append(fpr)
    print "HOC feature extraction time for one fold: ", model_time
    time_fold_data.append(model_time)

  return accuracy_fold_data, false_positive_rate_fold_data, time_fold_data

#Machine learning pipeline for prediction using HOC features
#Feature extraction to model training and usage
#Input: adjacency matrix (data)
#       Name of dataset to use
#       Maximum cycle order to consider
#       Number of folds for k-fold cross validation (default 10 like in the paper)
#       Number of features to use (to test whether classifier is actually learning)
#Output: average accuracy, false positive rate across folds
def hoc_learning_pipeline(adj_matrix, max_cycle_order, num_folds=10, num_features=-1):
  #Split into folds
  unique_edge_list = pipeline.get_unique_edges(adj_matrix)
  data_folds = pipeline.kfold_CV_split(unique_edge_list, num_folds)

  #Perform k-fold cross validation
  acc_fold_data, fpr_fold_data, time_fold_data = kfold_CV(adj_matrix, data_folds, max_cycle_order, num_features)
  avg_acc = sum(acc_fold_data) / float(len(acc_fold_data))
  avg_fpr = sum(fpr_fold_data) / float(len(fpr_fold_data))
  avg_time = sum(time_fold_data) / float(len(time_fold_data))
  acc_stderr = analytics.error_width(analytics.sample_std(acc_fold_data), num_folds)
  fpr_stderr = analytics.error_width(analytics.sample_std(fpr_fold_data), num_folds)
  time_stderr = analytics.error_width(analytics.sample_std(time_fold_data), num_folds)
  return avg_acc, acc_stderr, avg_fpr, fpr_stderr, avg_time, time_stderr

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
  avg_acc, stderr_acc, avg_fpr, stderr_fpr, avg_time, stderr_time = hoc_learning_pipeline(adj_matrix, max_cycle_order, num_folds, num_features)
  print("Accuracy: average %f with standard error %f" % (avg_acc, stderr_acc))
  print("False positive rate: average %f with standard error %f" % (avg_fpr, stderr_fpr))
  print("Model running time: average %f with standard error %f" % (avg_time, stderr_time))

