import pandas as pd
import numpy as np
import glob

def read_data(folder_path):
  '''
    Input --> folder_path - path of the folder which 
                            contains all csv files 
    
    All the names of different are converted to lower 
    case, and finally all datframes are merged to one.
    
    Output --> result = result is the combined data
  '''
  files = glob.glob(folder_path + "/*.csv")  # list of all .csv file paths
  df1 = pd.read_csv(files[0])  
  df1.drop('race',1,inplace = True)
  df2 = pd.read_csv(files[1])
  df2.drop('race',1,inplace = True)
  df3 = pd.read_csv(files[2])
  
  df3['name'] = df3['name'].apply(lambda x: str(x).lower())  # converting to lower case
  df3['gender'] = df3['gender'].apply(lambda x: str(x).lower())
  frames = [df1, df2,df3]
  result = pd.concat(frames)  # concatinating all dataframes
  result = result.sample(frac=1)  # data shuffling
  return result

def reverse_data(data):
  '''
    Input --> data - takes prepared dataframe as input
    
    It identifies those name which has 2 or more words
    and training data is increased by reversing their names.
    
    Output --> new_data - data which contains reversed names
                          (final data to be used further)
  '''
  data = data[~data.name.str.contains("@")] # removing data having @
  
  multi_word = data[data.name.str.contains(" ")]
  multi_word['name'] = multi_word['name'].apply(len_detect)  # new data with reversed names
  new_data = pd.concat([data, multi_word], axis=0)  # merged with training data
  return new_data
