from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from imblearn.datasets import make_imbalance
import numpy as np
from Bilstm import Bilstm,Bilstm_THS,X,Y
import sklearn
from sklearn.model_selection import train_test_split

vocab_size=100
#input_dim = 75
embedding_dim = 100
hidden_dim = 256
padding_idx=0.5
output_dim = 32
n_layer = 2
bidirectional = True
DROPOUT = 0.5
n_classes=2
batch_size=32
dropout=0.5

model = Bilstm( vocab_size,max_num_hidden_layers,embedding_dim, padding_idx, static=True, n_classes,n_layer,
                  dropout, batch_size)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)

for i in range(len(X_train)):
  model.partial_fit(np.asarray([X_train[i, :]]), np.asarray([y_train[i]]))
  
  if i % 1000 == 0:
    predictions = model.predict(X_test)
    print("Online Accuracy: {}".format(balanced_accuracy_score(y_test, predictions)))
