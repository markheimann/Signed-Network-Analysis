# Signed-Network-Analysis

This is a replication study of the paper

Chiang, Kai-Yang, et al. "Prediction and clustering in signed networks: a local to global perspective." The Journal of Machine Learning Research 15.1 (2014): 1177-1213.

Originally for an artificial intelligence course project at the University of Michigan, Winter 2016.  Hopefully it will be useful for further exploration in signed network analysis research (and other problems in low-rank matrix completion, matrix factorization, and spectral clustering...all that good stuff)

To use: 

- Install any needed software requirements (see requirements.txt)

You can use simulated data (see running instructions) or download and use real signed network data as part of the Stanford Network Analysis Project (SNAP).

Real data: 
- Download datasets of signed networks: http://snap.stanford.edu/data/index.html (go to "Signed networks": we offer code to preprocess Epinions, Wikipedia (elections) and Slashdot datasets).

- Preprocess data to save its signed adjacency matrix.  Code to preprocess the data in the raw form provided by Stanford is available in preprocess_ES.py (Epinions and Slashdot) and preprocess_W.py (Wikipedia)

- We created a directory known as Preprocessed Data and saved the Wikipedia dataset under the name "wiki_elections_csr.npy".  Do the same or change the relevant parts of the code." 

Running:

-Run the file "experiments.py" to see the methods in action.  At the top of the file, boolean values are set to indicate which methods you want to use.  Change these as desired.  You can also select whether to use simulated or real data.

Further usage:

-Check individual files (and methods) to see how the algorithms themselves work.  

The main algorithms are: 
MOI (measure of imbalance): predicts signs for edges according to principles of balance theory from sociology
HOC (higher-order cycles): learns a supervised learning algorithm on top of features extracted from network to predict signs of edges
SVP: uses singular value projection to complete the adjacency matrix of the signed network for link prediction.
Matrix factorization: includes stochastic gradient descent (NOTE: doesn't work very well in our experiments. If you have suggestions, submit a pull request) and alternating least squares to solve a matrix factorization problem to complete the adjacency matrix of the signed network for link prediction.
simulate_network allows you to simulate incomplete signed network data by sampling uniformly at random from a signed complete network with size (and number of clusters of positive edges) you choose.  

See paper for details.  

Files with these terms in their names implement these methods (or helper methods).  There are also pipelines to implement more general machine learning functionality, including ml_pipeline.py (implements functionality used k-fold-cross validation framework). analytics.py implements a few methods to calculate standard errors and such.  Finally, tests.py implements some unit tests. 

cross-validation with MOI takes place in local_moi.py (using tools from ml_pipeline.py)
cross-validation with HOC takes place in hoc_prediction.py (using tools from ml_pipeline.py)
cross-validation with MOI takes place in matrix_prediction.py (using tools from ml_pipeline.py)


