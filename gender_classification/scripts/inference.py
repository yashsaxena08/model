import pickle
from data_cleaning import clean_data
from embedding import pre_embed
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import sequence
from keras.models import load_model
import json

with open('char_index.json','r') as file:
  char_index=json.load(out_file)



def predict(names, model_path='my_model.h5'):
  '''
    Input --> names: takes a list of names all should be in lower case
    
    Load saved model and create embeddings for input names and prints a 
    dictionary having male/female scores.
    
    Output --> pred: return array having score of Male/Female.
  '''
  model = pickle.load(open(model_path, 'rb'))
  names = clean_data(names)
  embeddings=[]
  for i in names:
    res = []
    for j in i:
      res.append(char_index[j])
    embeddings.append(res)

  padded_embedding = sequence.pad_sequences(embeddings, maxlen=40, value=98, padding = 'post')  # padded embeddings
  pred=model.predict(np.asarray(padded_embedding))
  results = pd.DataFrame()
  results['name'] = names
  results['male score'] = pred[:,0]
  results['female score'] = pred[:,1]
  print(results)  # prints dataframe having mal/female score with their names
  return pred
