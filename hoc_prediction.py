#Use local method (HOC) for sign prediction in signed networks
#Based on Chiang et. al, 2014

import numpy as np
import cPickle
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import norm
from sklearn.linear_model import LogisticRegression
import hoc_edge_features as hoc
import ml_pipeline as pipeline
import random, os

#Perform cross validation, testing on one fold and training on the rest
#Input: indices of data points in each folds
#       Features, labels to learn from
#Output: average test accuracy, false positive rate
def kfold_CV(folds, features, labels):
  num_folds = len(folds)
  accuracy = 0
  false_positive_rate = 0
  for fold_index in range(num_folds):
    print("Fold %d:" % (fold_index + 1))

    #get data
    train_points = pipeline.join_folds(folds, fold_index)
    test_points = folds[fold_index]   

    #get features and labels corresponding to each data point
    train_data = np.asarray([features[(datum[0], datum[1])] for datum in train_points])
    train_labels = np.asarray([labels[datum] for datum in train_points])
    test_data = np.asarray([features[datum] for datum in test_points])
    test_labels = np.asarray([labels[datum] for datum in test_points])

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
  features_dict, labels_dict = hoc.extract_edge_features(adj_matrix, dataset_name, max_cycle_order, dataset_name)
  print "number of features calculated: ", len(features_dict[features_dict.keys()[0]])

  #TODO: without this line classifier learns perfectly (not supposed to happen)
  #(figure out why)
  for key in features_dict.keys():
    random.shuffle(features_dict[key])

  #completely randomize the features
  #NOTE: with this test, classifier should just predict mode label
  if num_features == 0:
    for key in features_dict.keys():
      features_dict[key] = list(np.random.random(len(features_dict[key])))

  #choose only a subset of the features to learn from
  #note: fewer features (e.g. 4) --> classifier always predicts mode label
  if num_features > 0:
    for key in features_dict.keys():
      random.shuffle(features_dict[key])
      features_dict[key] = features_dict[key][:num_features] #choose a subset of features at random

  #Split into folds
  data_folds = pipeline.kfold_CV_split(features_dict.keys(), num_folds)

  #Perform k-fold cross validation
  avg_accuracy, avg_false_positive_rate = kfold_CV(data_folds, features_dict, labels_dict)
  return avg_accuracy, avg_false_positive_rate

if __name__ == "__main__":
  #data_file_name = "Preprocessed Data/wiki_elections_csr.npy"
  #dataset_name = "wikipedia"
  #data_file_name = "Preprocessed Data/Slashdot090221_csr.npy"
  #dataset_name = "slashdot"
  #data_file_name = "Preprocessed Data/epinions_csr.npy"
  #dataset_name = "epinions"
  data_file_name = "Preprocessed Data/small_network.npy"
  dataset_name = "small"

  if not os.path.exists(data_file_name):
    raise ValueError("invalid path for data file")
  adj_matrix = np.load(data_file_name).item()
  max_cycle_order = 2 #TODO is this equivalent to l=5 or l=6 in their work?

  num_folds = 10
  avg_accuracy, avg_false_positive_rate = hoc_learning_pipeline(adj_matrix, dataset_name, max_cycle_order, num_folds)
  print "Average accuracy: ", avg_accuracy
  print "Average false positive rate: ", avg_false_positive_rate

