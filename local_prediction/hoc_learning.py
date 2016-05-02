#Contains methods for sign prediction in signed networks
#Based on Chiang et. al, 2014

import numpy as np
import cPickle
from scipy.linalg import norm
from sklearn.linear_model import LogisticRegression
import hoc_edge_features as hoc
import random

#Split data into folds for cross-validation
#Input: graph data, labels (dictionaries consisting of edge:features or edge:label)
#  Number of folds (k in k-fold cross validation)
#Output: k disjoint sets of vertices whose 
#union is the set of all edges (nonzero entries in matrix)
#TODO Returns matrix of each fold's features, labels
def kfold_CV_split(data, num_folds=10):
  data_points = data# data.keys() #same as label keys
  random.shuffle(data_points)
  fold_size = len(data_points)/num_folds
  folds = list() #data points (edges) in each fold
  for fold_index in range(num_folds): #append evenly sized folds of data
    if fold_index == num_folds - 1: #last fold--append remaining data
      folds.append(data_points[fold_size*fold_index:])
    else:
      folds.append(data_points[fold_size*fold_index:fold_size*(fold_index + 1)])
  return folds

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
    train_points = join_folds(folds, fold_index)
    test_points = folds[fold_index]

    #check: make sure sets are disjoint
    train_set_points = set(train_points)
    test_set_points = set(test_points)
    print "size of train set: ", len(train_points), len(train_set_points)
    print "size of test set: ", len(test_points), len(test_set_points)
    print "train and test set intersection size: ", 
    print len(train_set_points.intersection(test_set_points))
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
    acc, fpr = evaluate(test_preds, test_labels)
    accuracy += acc
    false_positive_rate += fpr

  accuracy = accuracy / num_folds
  false_positive_rate = false_positive_rate/num_folds
  return accuracy, false_positive_rate

#Join folds to construct training dataset
#Input: all folds
# Index of fold to leave out
#Output: List of training points (all other folds)
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

#given test predictions and labels, evaluate metrics like accuracy
def evaluate(predictions, labels):
  print("Predictions: avg %f" % np.mean(predictions))
  print("Labels: avg %f" % np.mean(labels))
  accuracy = np.mean(predictions == labels)
  print("Accuracy: %f" % accuracy)
  
  #false positives: prediction 1 but actual label -1
  num_false_positives = np.sum(predictions == labels + 2)
  print "number of false positives: ", num_false_positives
  #test predictions and labels both -1
  num_true_negatives = np.sum(np.logical_and(predictions == -1,labels == -1))
  false_positive_rate = 0
  try:
    false_positive_rate = float(num_false_positives) / (num_false_positives + num_true_negatives)
  except ZeroDivisionError:
    print "OK...so no false positives and no true negatives? hmmm..."
  print("False positive rate: %f" % false_positive_rate)
  return accuracy, false_positive_rate


#Machine learning pipeline
#Feature extraction to model training and usage
def ml_pipeline(adj_matrix, dataset_name, max_cycle_order, num_folds=10):
  #Get data
  features_dict, labels_dict = hoc.extract_edge_features(adj_matrix, max_cycle_order, dataset_name)
  print "number of features calculated: ", len(features_dict[features_dict.keys()[0]])

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
  data_folds = kfold_CV_split(features_dict.keys(), num_folds)

  #Perform k-fold cross validation
  avg_accuracy, avg_false_positive_rate = kfold_CV(data_folds, features_dict, labels_dict)
  return avg_accuracy, avg_false_positive_rate

if __name__ == "__main__":
  data_file_name = "Preprocessed Data/wiki_elections_csr.npy"
  dataset_name = "wikipedia"
  #data_file_name = "Preprocessed Data/Slashdot090221_csr.npy"
  #dataset_name = "slashdot"
  #data_file_name = "Preprocessed Data/epinions_csr.npy"
  #dataset_name = "epinions"
  adj_matrix = np.load(data_file_name).item()
  max_cycle_order = 2 #TODO is this equivalent to l=5 or l=6 in their work?

  num_folds = 10
  avg_accuracy, avg_false_positive_rate = ml_pipeline(adj_matrix, dataset_name, max_cycle_order, num_folds)
  print "Average accuracy: ", avg_accuracy
  print "Average false positive rate: ", avg_false_positive_rate

