import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class labelEnc():
    """
    Label Encoding
    """
    def __init__(self, categorical_features):
        self.key = categorical_features
        self.enc = [LabelEncoder() for c in categorical_features]
      
    def fit(self,X, y=None):
        y = np.array(X)
        self.categorical_feat_classes = {}
        for i in range(len(self.key)):
          c = self.key[i]
          self.enc[i] = self.enc[i].fit(y[:,c])
          self.categorical_feat_classes[c] = self.enc[i].classes_
        return self
      
    def transform(self,X):
        y = np.array(X)
        for i in range(len(self.key)):
          c = self.key[i]
          y[:,c] = self.enc[i].transform(y[:,c])
        return y
      
    def fit_transform(self,X, y=None):
        self.fit(X,y)
        y = np.array(X)
        for i in range(len(self.key)):
          c = self.key[i]
          y[:,c] = self.enc[i].transform(y[:,c])
        return y

class oneHotEnc():
    """
    One Hot Encoding
    """
    def __init__(self, categorical_features, fit_onehot):
        self.key = categorical_features
        self.enc = fit_onehot.set_params(categorical_features=categorical_features)

    def fit(self, X, y=None):
        #global categorical_onehot_idx,categorical_onehot_nval 
        self.enc = self.enc.fit(X,y)
        categorical_onehot_idx = self.enc.feature_indices_
        categorical_onehot_nval = self.enc.n_values_
        self.categorical_onehot_idx = categorical_onehot_idx
        self.categorical_onehot_nval = categorical_onehot_nval
        return self

    def transform(self, X):
        y = self.enc.transform(X)
        y = y if type(y)==np.ndarray else y.toarray()
        return y
      
    def fit_transform(self,X, y=None):
        y = self.fit(X,y).transform(X)
        y = y if type(y)==np.ndarray else y.toarray()
        return y