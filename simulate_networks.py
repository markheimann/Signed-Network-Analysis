#Simulate signed networks
#Based on Chiang et. al, 2014

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import norm

#Create a complete k-weakly balanced network
#   within cluster edges positive, between cluster edges negative
#In paper set k = 5 with 100 nodes per cluster
#Input: k (number of clusters)
#   how many nodes to each cluster
#Output: signed network (complete graph 
#with signs according to k-weakly balanced definition)
def simulate_full_weakly_balanced_network(num_clusters, cluster_size):
  num_nodes = num_clusters * cluster_size

  #will make block diagonal matrix (clusters on blocks on diagonal)
  sign_matrix = np.zeros((num_nodes, num_nodes)) - 1 #off entries -1
  for cluster_count in range(num_clusters):
    #pick out edges within cluster and make them 1s 
    start_ones = cluster_count*cluster_size #index to start
    end_ones = (cluster_count+1)*cluster_size #index to end at
    sign_matrix[start_ones:end_ones,start_ones:end_ones] = 1
  return csr_matrix(sign_matrix)

#Input: (full) network to sample from
#   sparsity level (percentage of edges to sample)
#   noise level (probability sample edge sign is flipped)
#   sampling process (uniform or power law)
#Output: noisy sampled network
#TODO: write official tests
#TODO: power law sampling
def sample_from_full_network(full_network, 
                              sparsity_level, 
                              noise_prob, 
                              sampling_process = "uniform"):

  #will fill in with correct entries
  sampled_network = np.zeros(full_network.shape).flatten() #sampling from 1D array is easier
  network_size = full_network.size
  num_samples = sparsity_level*network_size #number of samples to take from network

  #Get probability with which to sample each entry
  probs = None #uniform (default choice)
  if sampling_process == "power_law":
    pass

  #Sample indices
  samples = np.random.choice(np.arange(network_size), num_samples, replace=False, p = probs)
  
  #Determine which indices to sample noisily (with wrong sign)
  noise = np.random.random(num_samples)
  true_entries = samples[np.where(noise > noise_prob)] 
  noisy_entries = samples[np.where(noise <= noise_prob)]

  #decide which entries we will sample and how
  TRUE_ENTRY = 2.0 #entries with this value will be sampled without noise
  NOISY_ENTRY = -2.0 #entries with this value will be sampled noisily (sign reversed)
  sampled_network[true_entries] = TRUE_ENTRY
  sampled_network[noisy_entries] = NOISY_ENTRY

  sampled_network = sampled_network.reshape(full_network.shape) #so we can index from full network

  #TODO figure out indexing from sparse array better instead of just converting to dense numpy array
  #select sample indices and then get their (real or noisy) values
  sample_true_rows, sample_true_cols = np.where(sampled_network == TRUE_ENTRY)
  sampled_network[sample_true_rows, sample_true_cols] = np.asarray(full_network.todense())[sample_true_rows, sample_true_cols]
  
  sample_noisy_rows, sample_noisy_cols = np.where(sampled_network == NOISY_ENTRY)
  sampled_network[sample_noisy_rows, sample_noisy_cols] = -1*np.asarray(full_network.todense())[sample_noisy_rows, sample_noisy_cols]
  return sampled_network

if __name__ == "__main__":
  sim_full_network = simulate_full_weakly_balanced_network(3,2)
  sim_partial_network = sample_from_full_network(sim_full_network,0.5,0.5)
  print "Full network: "
  print sim_full_network.todense()
  print "Partial network: "
  print sim_partial_network
  print sim_full_network == sim_partial_network
  print sim_full_network == -1 * sim_partial_network

