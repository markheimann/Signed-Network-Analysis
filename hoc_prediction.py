#Contains methods for sign prediction in signed networks
#Based on Chiang et. al, 2014

import numpy as np
import cPickle
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import norm
from sklearn.linear_model import LogisticRegression
import hoc_edge_features as hoc
import ml_pipeline as pipeline
import random

#Perform cross validation, testing on one fold and training on the rest
#Input: indices of data points in each folds
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

    #WRITETEST
    #check: make sure sets are disjoint
    train_set_points = set(train_points)
    test_set_points = set(test_points)
    print "size of train set: ", len(train_points), len(train_set_points)
    print "size of test set: ", len(test_points), len(test_set_points)
    print "train and test set intersection size: ", 
    print len(train_set_points.intersection(test_set_points))

    #WRITETEST
    '''
    print "train-test intersection including reverse edges: "
    all_train_edge_set = set(train_points + [point[::-1] for point in train_points])
    all_test_edge_set = set(test_points + [point[::-1] for point in test_points])
    print len(all_train_edge_set.intersection(all_test_edge_set))

    #ok so no valueerror--integrity of train, test set seems ok
    for point in test_points:
      if point in train_set_points or point[::-1] in train_set_points:
        raise ValueError("test edge in training set")
    '''

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
def hoc_learning_pipeline(adj_matrix, dataset_name, max_cycle_order, num_folds=10):
  #Get data
  features_dict, labels_dict = hoc.extract_edge_features(adj_matrix, max_cycle_order, dataset_name)
  print "number of features calculated: ", len(features_dict[features_dict.keys()[0]])

  #WRITETEST(?)
  #completely randomize the features
  #NOTE: with this test, classifier just predicts mode label
  #for key in features_dict.keys():
  #  features_dict[key] = list(np.random.random(len(features_dict[key])))

  #choose only a subset of the features to learn from
  NUM_FEATURES = 14 #4: classifier almost always predicts mode label
  for key in features_dict.keys():
    random.shuffle(features_dict[key])
    features_dict[key] = features_dict[key][:NUM_FEATURES] #choose a subset of features at random

  #Split into folds
  data_folds = pipeline.kfold_CV_split(features_dict.keys(), num_folds)

  #Perform k-fold cross validation
  avg_accuracy, avg_false_positive_rate = kfold_CV(data_folds, features_dict, labels_dict)
  return avg_accuracy, avg_false_positive_rate

if __name__ == "__main__":
  #EXCEPTIONHANDLING make sure this file exists
  data_file_name = "Preprocessed Data/wiki_elections_csr.npy"
  dataset_name = "wikipedia"
  #data_file_name = "Preprocessed Data/Slashdot090221_csr.npy"
  #dataset_name = "slashdot"
  #data_file_name = "Preprocessed Data/epinions_csr.npy"
  #dataset_name = "epinions"
  adj_matrix = np.load(data_file_name).item()
  max_cycle_order = 2 #TODO is this equivalent to l=5 or l=6 in their work?

  num_folds = 10
  avg_accuracy, avg_false_positive_rate = hoc_learning_pipeline(adj_matrix, dataset_name, max_cycle_order, num_folds)
  print "Average accuracy: ", avg_accuracy
  print "Average false positive rate: ", avg_false_positive_rate

