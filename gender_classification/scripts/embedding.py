from tensorflow.keras.preprocessing import sequence
import numpy as np
from data_preprocessing import *

data = read_data()
new_data = reversed_data(data)

def pre_embed():
  '''
    Creates of vocab of having all the unique 
    characters and eturns char_index which 
    will be used further for embedding creation.
  '''
  names = new_data['name'].to_list()
  gender = new_data['gender'].to_list()
  vocab = set(' '.join([str(i) for i in names]))
  vocab.add('END')
  char_index = dict((c, i) for i, c in enumerate(vocab))  # creating unique indexing numbers
  return char_index, names

def embed_train_data():
  '''
    Uses different index value for every character
    from char_index and finally padded for the same
    shape and return train_x and train_Y
  '''
  char_index, names = pre_embed()
  x = []
  for i in names:
    res = []
    for j in i:
      res.append(char_index[j])
    x.append(res)
  x_train = sequence.pad_sequences(x, maxlen=40, value=96, padding = 'post')  # padding
  train_X = np.array(x_train)
  
  train_Y = []
  for i in gender:
      if i == 'm':
          train_Y.append([1,0])
      else:
          train_Y.append([0,1])
  train_Y = np.array(train_Y)
  
  return train_X, train_Y
