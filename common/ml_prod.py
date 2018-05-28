import xgboost as xgb
import pandas as pd
import numpy as np

## Save model
class ml_model():
  def __init__(self, pipeline_preprocess, model):
    self.pipeline_preprocess = pipeline_preprocess
    self.model = model
  
  def predict(self, x):
    y = scoreCreditPD(x, self.pipeline_preprocess, self.model)
    return y
  
def scoreCreditPD(data, pipeline_preprocess, model, label_encoder=None):
  X= data
  #X = X.drop(["tgl_akad_awal","target"], axis=1)
  if label_encoder != None:
    X = label_encoder.transform(X)
  X = pipeline_preprocess.transform(X)
  try:
    y = model.predict_proba(X)
  except (AttributeError):
    X1 = xgb.DMatrix(X)
    y1 = model.predict(X1)
    y0 = 1-y1
    y = np.array([y0,y1]).transpose()
  return y

def scoreCredit(x, label_encoder, pipeline_preprocess, model):
  y = scoreCreditPD(x, pipeline_preprocess, model, label_encoder=label_encoder)
  y = np.float64(y[:,1])
  return y

def scoreCreditJSON(data, label_encoder, pipeline_preprocess, model,dtype):
  data1 = pd.Series(data).to_frame().transpose()
  colexc = [c for c in data1.columns.values if not(c in dtype.index.values)]
  data1 = data1.drop(colexc, axis=1)
  for c in data1.columns.values:
    typ = dtype[c]
    data1[c] = data1[c].astype(typ)
    data1 = data1.loc[:,dtype.index.values]
  
  y = scoreCredit(data1, label_encoder, pipeline_preprocess, model)
  return y

class bst_models():
  
  def __init__(self,colname, score_method="mean"):
    self.model = []
    self.colname = colname
    self.score_method = score_method
  
  def add_model(self, model):
    self.model.append(model)
    return self
  
  def predict(self,X):
    d = xgb.DMatrix(X,feature_names=self.colname)
    preds=[]
    for m in self.model:
      pred = m.predict(d)
      preds.append(list(pred))
      
    K = len(preds)
    nrow = len(preds[0])
    self.K = K
    self.nrow = nrow
    pred = []
    for i in range(nrow):
      sum1=0
      for j in range(K):
        if self.score_method == "max":
          sum1 = max(sum1, preds[j][i])
        else:
          sum1+=preds[j][i]
      if self.score_method == "max":
        pred.append(sum1)
      else:
        pred.append(sum1/K)
      
    return pd.Series(pred)
  
  def get_importances(self):
    imp = dict()
    cols = self.model[0].feature_names
    for col in cols:
      imp1=0
      k1 = 0
      for j in range(self.K):
        try:
          imp1+=self.model[j].get_score()[col]
        except (KeyError):
          pass
        k1 += 1
      imp[col] = imp1 / k1
    imp = pd.Series(imp).sort_values(ascending=False)
    return imp
  
class skl_models():
  
  def __init__(self,colname, score_method="mean"):
    self.model = []
    self.colname = colname
    self.score_method = score_method
  
  def add_model(self, model):
    self.model.append(model)
    return self
  
  def predict(self,X):
    preds=[]
    for m in self.model:
      pred = m.predict_proba(X)[:,1]
      preds.append(list(pred))
      
    K = len(preds)
    nrow = len(preds[0])
    self.K = K
    self.nrow = nrow
    pred = []
    for i in range(nrow):
      sum1=0
      for j in range(K):
        if self.score_method == "max":
          sum1 = max(sum1, preds[j][i])
        else:
          sum1+=preds[j][i]
      if self.score_method == "max":
        pred.append(sum1)
      else:
        pred.append(sum1/K)
      
    return pd.Series(pred)
  
      