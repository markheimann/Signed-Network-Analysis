#Implement mDA (Chen et. al, 2012)
#Take in data and probability of corruption
#"Corrupt" data (but marginalize out the (expected) corruption) and learn a reconstruction specified by weights
import numpy as np
import msda

import data.simulate_networks as sim

if __name__ == "__main__":
	simulated = True
	real = False
	sparsity_level = 0.5

	adj_matrix = None
	if simulated:
		cluster_sizes = [100,200,300,400]
		sparsity_level = 0.01175
		noise_prob = 0
		print "creating adjacency matrix..."
		adj_matrix = sim.sample_network(cluster_sizes, sparsity_level, noise_prob)
		complete_network = sim.construct_full_network(cluster_sizes).todense()

	elif real:
		data_file_name = "data/Preprocessed Data/small_network.npy"
		#data_file_name = "data/Preprocessed Data/wiki_elections_csr.npy"
		try:
			adj_matrix = np.load(data_file_name).item()
		except Exception as e:
			raise ValueError("could not load adj matrix from file: ", e)

	adj_matrix = adj_matrix.todense() #for now use dense to match msda TODO write sparse mSDA
	prob_corruption = 0.2#1 - sparsity_level
	num_layers = 3
	mapping, representations = msda.mDA(adj_matrix, prob_corruption)
	print mapping.shape
	matrix_complet = np.sign(np.dot(mapping.T, adj_matrix))
	matrix_complet = matrix_complet.astype(int)
	matrix_complet = matrix_complet[:1000,:1000] #TODO why is matrix_complete naturally (1001,1000) -- because of bias...
	print matrix_complet[:10,:10]
	print adj_matrix[:10,:10]
	print complete_network[:10,:10]
	#print matrix_complet
	print "number of nonzero entries in original matrix: ", np.count_nonzero(adj_matrix)
	print "number of entries in original matrix recovered by msda: ", np.sum(matrix_complet == adj_matrix)
	print "number of entries in complete matrix recovered by msda: ", np.sum(matrix_complet == complete_network)
	print "false negatives in complete matrix: ", np.sum(matrix_complet == complete_network - 2) #matrix completion predicts -1 where should be 1
	print "false positives in complete matrix: ", np.sum(matrix_complet == complete_network + 2) #matrix completion predicts 1 where should be -1
	print "number of 1s in completed: ", np.sum(matrix_complet == 1)
	print "number of -1s in completed: ", np.sum(matrix_complet == -1)

	print np.asarray(matrix_complet).shape# == np.asarray(complete_network)
