#Contains test for the rest of the code

import unittest
import numpy as np
import scipy.sparse as sp

import matrix_completion.svp_sign_prediction as svp
import local_prediction.hoc_edge_features as hoc_features
import local_prediction.hoc_prediction as hoc_prediction
import data.preprocess_ES
import data.preprocess_W
import utils.ml_pipeline as pipeline
import data.simulate_networks as sim
import cluster.clustering

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
    max_cycle_order = 5
    mode = "test"
    hoc_features.extract_edge_features(adj_matrix, max_cycle_order, mode)

  #Test recursive algorithm to compute feature matrix products
  def test_compute_feature_products(self):
    matrix1 = sp.csr_matrix(np.asarray([[2,3],[4,5]]))
    matrix2 = sp.csr_matrix(np.asarray([[1,6],[3,2]]))
    matrix3 = sp.csr_matrix(np.asarray([[4,4],[5,2]]))
    matrix4 = sp.csr_matrix(np.asarray([[3,2],[8,-1]]))
    components = [matrix1, matrix2, matrix3, matrix4]
    products = hoc_features.compute_feature_products(components,3)

    #make sure first product is computed correctly for each length
    for length in products.keys():
      assert np.array_equal(products[length][0].todense(), (matrix1 ** length).todense())

  #test getting unique edges from a matrix
  #i.e. edge (i,j) and (j,i) not both in matrix
  def test_get_unique_edges(self):
    data_file_name = "Preprocessed Data/small_network.npy"
    adj_matrix = np.load(data_file_name).item()
    unique_edges = hoc_prediction.get_unique_edges(adj_matrix)
    unique_edge_set = set(unique_edges)
    for edge in unique_edges:
      #make sure either edge is symmetric
      #or its reverse is not in set
      assert edge == edge[::-1] or edge[::-1] not in unique_edge_set

  #tests with more subjective criteria to see what HOC classifier actually learns
  def test_hoc_learning(self):
    data_file_name = "Preprocessed Data/small_network.npy"
    adj_matrix = np.load(data_file_name).item()
    max_cycle_order = 5

    #should just predict mode label of 1 because features are random noise
    #so only positives and some are false
    avg_acc, avg_fpr = hoc_prediction.hoc_learning_pipeline(adj_matrix, 
                      max_cycle_order, num_folds = 5, num_features = 0)

#Test machine learning pipeline for generic k-fold cross validation (on graph data)
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

#Test global prediction methods: SVP, SGD, ALS
class test_global_prediction(unittest.TestCase):
  #Test SVP completion on small matrix
  def test_SVP(self):
    data_file_name = "Preprocessed Data/small_network.npy"
    adj_matrix = np.load(data_file_name).item()
    rank = 5
    tol = 1
    max_iter = 10
    step_size = 1
    matrix_complet = svp.sign_prediction_SVP(adj_matrix, rank, tol,
                                        max_iter, step_size, mode = "test")
    np_sol = np.asarray(matrix_complet.todense())
    prop_recovered = float(np.sum(adj_matrix == np_sol))/(adj_matrix.nnz)
    #matrix completion doesn't ruin original entries in small dataset
    #TODO: write more extensive tests where you test the completion part too
    assert prop_recovered == 1.0

class test_simulation(unittest.TestCase):
  #toy example for all tests
  cluster_sizes = [1,2,3,4]
  sim_full_network = sim.construct_full_network(cluster_sizes)
  network_size = sum(cluster_sizes)
  
  #make sure we're calculating the weakly-balanced complete matrix correctly
  def test_get_edge_sign(self):
    edge1 = sim.get_complete_balanced_network_edge_sign(self.cluster_sizes,(2,2))
    assert edge1 == 1 #should be at right edge of second cluster
    edge2 = sim.get_complete_balanced_network_edge_sign(self.cluster_sizes,(2,3))
    assert edge2 == -1 #should be just outside second cluster

  #basic test: creates the network we think in easiest case (no sparsity or noise)
  def test_sample_network_simulation(self):
    sparsity = 1
    noise_prob = 0
    sim_partial_network = sim.sample_network(self.cluster_sizes,sparsity,noise_prob)

    #without sparsity should completely recover network
    #and no noise either
    assert (self.sim_full_network != sim_partial_network).nnz == 0

  #test to make sure generating networks of desired sparsity
  def test_sample_network_sparse_quantity(self):
    #with sparsity should in expectation recover a lot of the network
    sparsity = 0.5
    noise_prob = 0
    sim_partial_network = sim.sample_network(self.cluster_sizes,sparsity,noise_prob, symmetric=False)
    expected_num_edges = self.network_size**2 * sparsity

    #Probabilistic test: MAY FAIL WITH CORRECT BEHAVIOR (BUT W/ LOW PROBABILITY)
    #by a Chernoff bound, this should deviate from expected number of edges
    #by more than t*(network_size/2)
    #with probability 2*e^(-t^2/2)
    #here set t = 4, so this tests passes with probability 1 - 2*e^-8
    actual_num_edges = sim_partial_network.nnz
    print expected_num_edges, actual_num_edges
    assert abs(actual_num_edges - expected_num_edges) <= 20

  #test to make sure noise is flipping edges
  def test_sample_network_noise(self):
    #with complete noise should recover opposite of matrix
    #could write another probabilistic test for the case of partial noise
    #but this is OK
    sparsity = 1
    noise_prob = 1
    sim_partial_network = sim.sample_network(self.cluster_sizes,sparsity,noise_prob)

    #should recover opposite of matrix
    assert (self.sim_full_network == sim_partial_network).nnz == 0

class test_clustering(unittest.TestCase):
  cluster_sizes = [3,4,5]
  sparsity = 0.5
  noise_prob = 0.1
  network_params = (cluster_sizes, sparsity, noise_prob)
  mode = "test"

  #test that signed laplacian is computed correctly
  #for a toy example we construct
  def test_signed_laplacian(self):
    cluster_sizes = [1,2,3,4]
    network_size = sum(cluster_sizes)
    full_network = sim.construct_full_network(cluster_sizes)
    signed_laplacian = clustering.signed_laplacian(full_network)

    for row in range(network_size):
      for col in range(network_size):
        if row == col:
          #since this is full matrix, absolute degree will be network_size
          #and all diagonal entries of adj matrix are 1
          assert signed_laplacian[row,col] == network_size - 2
        else:
          #since off diagonal entries of degree matrix are 0
          #so subtracting full network entries from 0
          assert signed_laplacian[row,col] == -full_network[row,col]

  #test that signed laplacian runs smoothly
  #clustering tests inline with clustering method
  # (right number of eigenvalues, clusters, etc.)
  #runs additional tests to make sure signed laplacian is PSD
  def test_signed_laplacian_clustering(self):
    method = "signed laplacian"
    completion_alg = None
    completion_params = None

    clustering_params = (self.cluster_sizes, method, completion_alg, completion_params, self.mode)
    cluster_acc = clustering.clustering_pipeline(self.network_params, clustering_params)

  #test that clustering (and matrix completion) runs smoothly
  def test_matrix_completion_clustering(self):
    method = "matrix completion"
    completion_alg = "svp"
    rank = 10
    tol = 1
    max_iter = 10
    step_size = 1
    completion_params = (rank, tol, max_iter, step_size)

    clustering_params = (self.cluster_sizes, method, completion_alg, completion_params, self.mode)
    cluster_acc = clustering.clustering_pipeline(self.network_params, clustering_params)


if __name__ == "__main__":
  print "Note: some tests are probabilistic and may ",
  print "fail with correct behavior (but only with low probability)"
  print
  print "Running tests..."

  unittest.main()