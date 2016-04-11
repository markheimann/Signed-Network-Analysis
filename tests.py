#Contains test for the rest of the code

import unittest
import numpy as np
from scipy.sparse import csr_matrix

import svp_sign_prediction
import hoc_edge_features as hoc_features
import hoc_prediction
import preprocess_ES
import preprocess_W
import ml_pipeline as pipeline

class test_preprocess(unittest.TestCase):
  #test preprocessing of wikipedia dataset
  def test_wikipedia_preprocess(self):
    preprocess_W.preprocess("test")

  #test preprocessing of Epinions/Slashdot
  def test_ES_preprocess(self):
    preprocess_ES.preprocess("test")


class test_hoc(unittest.TestCase):
  def test_extract_edge_features(self):
    data_file_name = "Preprocessed Data/small_network.npy"
    adj_matrix = np.load(data_file_name).item()
    network_name = "small"
    max_cycle_order = 3
    mode = "test"
    hoc_features.extract_edge_features(adj_matrix, network_name, max_cycle_order, mode)

  #Test recursive algorithm to compute feature matrix products
  def test_compute_feature_products(self):
    matrix1 = csr_matrix(np.asarray([[2,3],[4,5]]))
    matrix2 = csr_matrix(np.asarray([[1,6],[3,2]]))
    matrix3 = csr_matrix(np.asarray([[4,4],[5,2]]))
    matrix4 = csr_matrix(np.asarray([[3,2],[8,-1]]))
    components = [matrix1, matrix2, matrix3, matrix4]
    products = hoc_features.compute_feature_products(components,3)

    #make sure first product is computed correctly for each length
    for length in products.keys():
      assert np.array_equal(products[length][0].todense(), (matrix1 ** length).todense())

  def test_hoc_learning(self):
    data_file_name = "Preprocessed Data/small_network.npy"
    adj_matrix = np.load(data_file_name).item()
    max_cycle_order = 3
    network_name = "small_network"

    #should just predict mode label of 1 because features are random noise
    #so only positives and some are false
    avg_acc, avg_fpr = hoc_prediction.hoc_learning_pipeline(adj_matrix, 
                      network_name, max_cycle_order, num_folds = 5, num_features = 0)
    assert avg_fpr == 1.0 

    #should mostly predict mode label of 1 because few features to learn from
    #so high rate of false positives
    avg_acc, avg_fpr = hoc_prediction.hoc_learning_pipeline(adj_matrix, 
                       network_name, max_cycle_order, num_folds = 5, num_features = 3)
    assert avg_fpr >= 0.8


class test_ml_pipeline(unittest.TestCase):
  #Make sure dataset is divided up properly 
  #into mutually exclusive, collectively exhaustive folds
  def test_form_folds(self):
    folds = range(100)
    num_folds = 3 #Note 100 not divisible by 3 so not all folds equal size

    folds = pipeline.kfold_CV_split(folds, num_folds)
    unique_data = set() #make sure all unique data points are accounted for
    for fold in folds:
      fold_set = set(fold)
      prev_num_unique = len(unique_data) 
      unique_data = unique_data.union(fold_set)
      #check that all the points in the fold were unique
      assert len(unique_data) - prev_num_unique == len(fold_set)
    assert len(unique_data) == 100 #check that all the data is in some fold

  #Make sure train and test sets are formed properly from folds
  def test_join_folds(self):
    #form 3 folds of sample data points
    folds = [[(i,i) for i in range(20)], \
              [(i,i) for i in range(20,40)], \
              [(i,i) for i in range(40,60)]]
     
    num_folds = 3 #test 3-fold cross validation

    for fold_index in range(num_folds):
      train_points = pipeline.join_folds(folds, fold_index)
      test_points = folds[fold_index]
      #check to make sure train and test datasets are disjoint
      #note: since graph is symmetric, (i,j) and (j,i) are basically same edge
      #and have same features: so if (i,j) is in train, (j,i) can't be in test
      #so check after augmenting training and test set with all reverse edges
      all_train_edge_set = set(train_points + [point[::-1] for point in train_points])
      all_test_edge_set = set(test_points + [point[::-1] for point in test_points])
      assert len(all_train_edge_set.intersection(all_test_edge_set)) == 0

if __name__ == "__main__":
  unittest.main()