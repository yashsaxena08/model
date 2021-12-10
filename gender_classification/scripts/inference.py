import pickle
from data_cleaning import clean_data
from embedding import pre_embed

char_index = pre_embed()[0]



def predict(names, model_path = 'gender_classification/pickle_model/model (2).pkl'):
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

  padded_embedding = sequence.pad_sequences(X, maxlen=40, value=98, padding = 'post')  # padded embeddings
  pred=model.predict(np.asarray(padded_embedding))
  results = pd.DataFrame()
  results['name'] = names
  results['male score'] = pred[:,0]
  results['female score'] = pred[:,1]
  print(results)  # prints dataframe having mal/female score with their names
  return pred
