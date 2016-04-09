'''Preprocess Wikipedia election data'''
'''Take in text file of data, create and save adjacency matrix'''

import numpy as np, cPickle
from scipy.sparse import csr_matrix



FILE_PATH = "Raw Data/wikiElec.ElecBs3.txt"

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

  pairs = dict() #keep track of what label we assigned to which pair
  count_pair_occurrences = dict() #keep track of how many times each pair has been voted on

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

      #sometimes we've already seen this person vote on this person
      #keep track of when this occurs and what happens
      if (active_user_IDs[voter_orig_ID], active_user_IDs[nominee_orig_ID]) in pairs:
        num_occurrences = count_pair_occurrences[(active_user_IDs[voter_orig_ID], active_user_IDs[nominee_orig_ID])]
        print("seen pair (%d, %d) %d times before; its label was " % (active_user_IDs[voter_orig_ID], active_user_IDs[nominee_orig_ID], num_occurrences)),
        print pairs[(active_user_IDs[voter_orig_ID], active_user_IDs[nominee_orig_ID])]
        count_pair_occurrences[(active_user_IDs[voter_orig_ID], active_user_IDs[nominee_orig_ID])] = num_occurrences + 1
      else:
        pairs[(active_user_IDs[voter_orig_ID], active_user_IDs[nominee_orig_ID])] = result
        count_pair_occurrences[(active_user_IDs[voter_orig_ID], active_user_IDs[nominee_orig_ID])] = 1
      labels.append(result)

  #Make a (square) adjacency matrix the size of the number of people 
  #as given by ID

  max_id = len(active_user_IDs.values())

  #Create in sparse row-major format
  #data_matrix = csr_matrix((np.array(labels), (np.array(from_data), 
  #             np.array(to_data)) ), shape=(max_id, max_id))
  data = np.array(labels)
  print np.min(data), np.max(data)
  row_ind = np.array(from_data)
  col_ind = np.array(to_data)
  M = max_id
  N = max_id
  data_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(M, N))

  #Sanity checks
  print "Number of unique users: ", max_id
  print "Number of edges: ", data_matrix.getnnz()
  print "Matrix shape: ", data_matrix.get_shape()

  print "max number of times a relationship was voted on: ", max(count_pair_occurrences.values())
  print "min and max values of data matrix: ", data_matrix.min(), data_matrix.max()
  print "taking signs (equivalently, the mode sign when same relationship was signed multiple times)"
  data_matrix = data_matrix.sign()
  print "min and max values of data matrix: ", data_matrix.min(), data_matrix.max()
  print "data matrix is symmetric?", (data_matrix != data_matrix.transpose()).nnz == 0
  if (data_matrix != data_matrix.transpose()).nnz > 0: #data matrix is not symmetric
    data_matrix = (data_matrix + data_matrix.transpose()).sign()
    print "fix"
    print "data matrix is symmetric?", (data_matrix != data_matrix.transpose()).nnz == 0


  #Save data
  np.save("Preprocessed Data/wiki_elections_csr", data_matrix)

