import pandas
import pickle
from data_cleaning import clean_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dense, Activation, Dropout, LSTM, Bidirectional

def build():
  '''
  Build a sequential LSTM model with an embedding layer
  with activation function as softmax and returns model.
  '''
  print('-----Model Building-----')
  model = Sequential()
  model.add(Embedding(97, 100, input_length=40))  # embedding layer 
  model.add(LSTM(512, return_sequences=True))  # LSTM layer 1
  model.add(Dropout(0.2))  # dropout layer
  model.add(LSTM(512, return_sequences=False))  # LSTM layer 2
  model.add(Dropout(0.2))
  model.add(Dense(2))
  model.add(Activation('softmax'))  
  model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
  print(model.summary())  # model summary
  return model

model = build()

def model_fit(train_X, train_Y, pkl_file = True):
  '''
    Input --> train_X, train_Y: training data for model which can be obtained
                                from embedding.py
    Fits the model and dump it to a pickle file
    
    Output --> model: trained model
  '''
  print('=======Training Model=======')
  batch_size=1000
  model.fit(train_X, train_Y,batch_size=batch_size,epochs=10,validation_split=0.3)
  model.fit(train_X, train_Y,batch_size=batch_size,epochs=5,validation_split=0.3)
  if pkl_file:
    pickle.dump(model, open('gender_classification/pickle_model/model.pkl', 'wb'))  # dump to pickle file
  return model
