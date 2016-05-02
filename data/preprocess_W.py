'''Preprocess Wikipedia election data'''
'''Take in text file of data, create and save adjacency matrix'''

import numpy as np, cPickle
import scipy.sparse as sp



FILE_PATH = "Raw Data/wikiElec.ElecBs3.txt"

#Preprocess wikipedia data
#Input: mode to run in (test or not)
#Action: saves signed adjacency matrix of wikipedia data to file
#        also saves signed adjacency matrix of small network
#        (first 100 people of wikipedia) to file
def preprocess(mode = "normal"):

  with open(FILE_PATH, "rb") as data_file:
    data_lines = data_file.readlines()

    #Save components of data in three lists kept in synchrony
    from_data = list()
    to_data = list()
    labels = list()

    #not all users participate
    #we don't care about those who don't so want to ignore them
    #make our own IDs for each unique user
    active_user_IDs = dict()

    active_ID = 0

    #Data format:
    #nominee line U <user (nominee) ID> <username> (we want ID)
    #vote line V <outcome> <user (voter) ID> <time> <username>
    nominee_orig_ID = "" #ID of current nominee ("to" vertex)
    for line in data_lines: 
      if line.startswith("U"): #user being nominated
        nominee_orig_ID = int(line.split()[1]) #update for a new user
      elif line.startswith("V"): #vote on the current user
        #get info
        info = line.split()
        result = int(info[1])
        voter_orig_ID = int(info[2]) #"from" vertex

        #create active IDs for nominee and voter if needed
        if voter_orig_ID not in active_user_IDs:
          active_user_IDs[voter_orig_ID] = active_ID
          active_ID += 1 #added another user
        if nominee_orig_ID not in active_user_IDs:
          active_user_IDs[nominee_orig_ID] = active_ID
          active_ID += 1 #added another user

        #add info
        #note: we added nominee and voter IDs to dict of active IDs
        #so they are guaranteed to exist
        from_data.append(active_user_IDs[voter_orig_ID])
        to_data.append(active_user_IDs[nominee_orig_ID])
        labels.append(result)

    #Make a (square) adjacency matrix the size of the number of people 
    #as given by ID

    max_id = len(active_user_IDs.values())

    #Create in sparse row-major format
    data = np.array(labels)
    print np.min(data), np.max(data)
    row_ind = np.array(from_data)
    col_ind = np.array(to_data)
    M = max_id
    N = max_id
    data_matrix = sp.csr_matrix((data, (row_ind, col_ind)), shape=(M, N)).sign()

    #Correction to make matrix symmetric (from paper)
    if (data_matrix != data_matrix.transpose()).nnz > 0: #data matrix is not symmetric
      data_matrix = (data_matrix + data_matrix.transpose()).sign()

    #Run tests
    if mode == "test":
      #make sure min value -1, max value 1
      assert data_matrix.min() == -1
      assert data_matrix.max() == 1

      #make sure data matrix is symmetric
      assert (data_matrix != data_matrix.transpose()).nnz == 0


    #Save data
    np.save("Preprocessed Data/wiki_elections_csr", data_matrix)
    np.save("Preprocessed Data/small_network", data_matrix[:100,:100])

if __name__ == "__main__":
  preprocess()

