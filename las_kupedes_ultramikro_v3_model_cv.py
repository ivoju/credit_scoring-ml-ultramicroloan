import pandas as pd

import matplotlib.pyplot as plt
import logging
import json
import re
from numpy.random import randint
from common import *
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.externals import joblib
from sklearn.utils import resample
import lime
from lime import lime_tabular
import seaborn as sns
import random
from imblearn.over_sampling import SMOTE
from IPython.display import Image

from common import *
from common.ml_dev import *
from common.ml_prod import *
import xgboost_explainer as xgb_exp

#data import
# import wget
import urllib

# regex
import re

#time
import pytz as tz
from datetime import datetime

#data explore
import numpy as np
np.random.seed(1337)

#support for reading excel files
#import xlrd

import keras

from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectFpr, SelectKBest, f_classif
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


import warnings
import itertools

from keras.callbacks import Callback

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

#%matplotlib inline
import matplotlib.pyplot as plt

#defaults
plt.rcParams['figure.figsize'] = (20.0, 20.0)
plt.rcParams.update({'font.size': 10})
plt.rcParams['xtick.major.pad']='5'
plt.rcParams['ytick.major.pad']='5'
plt.style.use('ggplot')

#%matplotlib inline
pd.options.display.html.table_schema = True
pd.options.display.max_columns = 99
%env JOBLIB_TEMP_FOLDER=/tmp


df = pd.read_csv("./data/las_dataset_kupedes_clean_v3_eda_v2.csv")
df = df.loc[df['plafond']<=25000000,:]
df = df.drop(labels=['status_bukti_kepemilikan_agunan','ratio_likuidasi_agunan_to_plafond','pemasaran'], axis=1)

### Split dataset
from sklearn.pipeline import Pipeline
#from imblearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline

dtype = df.dtypes.drop([
  'tgl_mulai','target',
#  'jenis_fasilitas','bln_mulai',
#  'suku_bunga','baru_perpanjangan','jabatan','bidang_pekerjaan','jenis_pekerjaan','region'
])
list_col = dtype.index.values.tolist()

X = df.loc[:, list_col]
Y = df['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
Y_train.reset_index(drop=True, inplace=True)
Y_test.reset_index(drop=True, inplace=True)
#X_train.apply(lambda x: x.count(), axis=0)


### Label Encoding
list_col_cat = [x for x in X_train.columns.values if ((X_train[x].dtypes==object) or (type(X_train[x].dtypes)==pd.core.dtypes.dtypes.CategoricalDtype))]
list_idx_cat = [i for i in range(len(X_train.columns.values)) if X_train.columns.values[i] in list_col_cat]
le = labelEnc(categorical_features=list_idx_cat)
le.fit(X)
categorical_feat_classes = le.categorical_feat_classes

### One Hot Encoding
onehotenc = oneHotEnc(categorical_features=list_idx_cat, fit_onehot=OneHotEncoder())

"""
tmp1 = le.fit(X_train,Y_train).transform(X_train)
tmp = onehotenc.fit(tmp1,Y_train).transform(tmp1)
tmp
"""

### Feature scaling
scaler = preprocessing.StandardScaler()

### Feature selection
N_feat = 42

warnings.simplefilter(action='ignore', category=(UserWarning,RuntimeWarning))
selector = SelectKBest(f_classif, N_feat)
#selector.fit(X_train, Y_train)
#top_indices = np.nan_to_num(selector.scores_).argsort()[-(N_feat):][::-1]
#selector.scores_[top_indices]
#colnames = X_train.columns[top_indices]
#colnames

"""
dtype,categorical_feat_classes,list_col_cat,list_idx_cat,categorical_onehot_idx,categorical_onehot_nval,colname = joblib.load('./model/las_kupedes_ultramikro_v3_var.sav')
#le,pipeline_preprocess = joblib.load('./model/las_kupedes_ultramikro_v3_preprocess.sav')
le,pipeline_preprocess = joblib.load('./model/las_kupedes_ultramikro_v3_preprocess_wo_scaler.sav')
"""

### Merge preprocess pipeline
X_train = le.transform(X_train)
X_test = le.transform(X_test)

# Preprocessing
pipeline = [
  # ('label_enc', le),
  ('one_hot_enc', onehotenc),
#  ('scale', scaler),
#  ('feature_selection', selector)
]
pipeline_preprocess = Pipeline(pipeline)

pipeline_preprocess.fit(X_train, Y_train)

# Get colname of new dataset features
onehotenc = pipeline_preprocess.steps[0][1]
categorical_onehot_idx = onehotenc.enc.feature_indices_
categorical_onehot_nval = onehotenc.enc.n_values_
colname = []
k=0
for i in categorical_onehot_idx[:-1]:
  feat = list_col_cat[k]
  for j in range(categorical_onehot_nval[k]):
    val = categorical_feat_classes[list_idx_cat[k]][j]
    colname.append("{}_{}".format(feat,val))
  k+=1
lcol_num = [x for x in dtype.index.values if not(x in list_col_cat)]
for i in lcol_num:
  colname.append("{}".format(i))

X_train = pipeline_preprocess.transform(X_train)
X_test = pipeline_preprocess.transform(X_test)

joblib.dump([dtype,categorical_feat_classes,list_col_cat,list_idx_cat,categorical_onehot_idx,categorical_onehot_nval,colname], './model/las_kupedes_ultramikro_v3_var.sav')
#joblib.dump([le,pipeline_preprocess], './model/las_kupedes_ultramikro_v3_preprocess.sav')
joblib.dump([le,pipeline_preprocess], './model/las_kupedes_ultramikro_v3_preprocess_wo_scaler.sav')

### Resampling unbalanced dataset
# (1) Over-sampling with SMOTE
def_ratio = 0.15
sm = SMOTE(random_state=42, ratio={0:Y_train.value_counts()[0],1:int(Y_train.value_counts()[0]*(def_ratio/(1-def_ratio)))})
sm.fit(X_train,Y_train)
X_train_upsampled, Y_train_upsampled = sm.sample(X_train, Y_train)
# (2) Class weight
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
sample_weight = compute_sample_weight(
  class_weight = {0:1,1:10},
  y = Y_train_upsampled
)

### CV - XGBoost

from sklearn.model_selection import KFold
K = 5
kf = KFold(n_splits = K, random_state = 3228, shuffle = True)

xgb_preds = []
bst = bst_models(colname=colname,score_method="max")

test = X_test; target_test = Y_test
for train_index, test_index in kf.split(X_train_upsampled):
    train = X_train_upsampled; target_train = Y_train_upsampled
    train_X, valid_X = train[train_index], train[test_index]
    train_y, valid_y = target_train[train_index], target_train[test_index]
    
    # Create logistic regression object
    d_train = xgb.DMatrix(train_X, label=train_y,feature_names=colname)
    d_valid = xgb.DMatrix(valid_X, label=valid_y,feature_names=colname)
    d_test  = xgb.DMatrix(test, label=target_test,feature_names=colname)

    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.75,
        'objective': 'binary:logistic',
        'silent': 1,
        'colsample_bytree': 0.9,
        'eval_metric': ['auc','logloss']
    }
    early_stopping = 25
    num_round = 100
    evallist  = [(d_train,'eval'), (d_valid,'train')]
    
    model = xgb.train(
      xgb_params, 
      d_train, 
      num_round,
      evallist,
      early_stopping_rounds=early_stopping
    )
    bst.add_model(model)

    
# Predict testing data
bst = joblib.load('./model/las_kupedes_ultramikro_v3_cv_xgb.sav')
y_train_pred = bst.predict(X_train)
y_test_pred = bst.predict(X_test)

compare = {
  'train': pd.DataFrame({'true':Y_train,'pred':y_train_pred}),
  'test': pd.DataFrame({'true':Y_test,'pred':y_test_pred})
}

# Model evaluation: KS
test_compare = compare['train']
N = test_compare.shape[0]
listidx = random.sample(test_compare.index.values.tolist(),  N)
disp = test_compare.loc[listidx,:]
sns.distplot(a=disp.loc[disp.true==0,'pred'], hist=True)
sns.distplot(a=disp.loc[disp.true==1,'pred'], hist=True)

result_ks = plot_ks(y_train_pred, Y_train)
result_ks.to_csv("out/las_kupedes_ultramikro_v3_cv_xgb_ks.csv",index=False)
result_ks
result_ks = plot_ks(y_test_pred, Y_test)
result_ks.to_csv("out/las_kupedes_ultramikro_v3_cv_xgb_ks_test.csv",index=False)
result_ks

stats.ks_2samp(
  compare['train'].loc[compare['train']['true']==0,'pred'], 
  compare['train'].loc[compare['train']['true']==1,'pred'])

#Make the plot canvas to write on to give it to the function  
from io import StringIO
import math
def write_stringIO(X):
    file = StringIO()
    for i,s in enumerate(X):
        file.write('{}\t{}\t{}\n'.format("label_A-%d" % i, "label_B-%d" % i, str(s)))
    return(file)
  
def make_hist(fh,ax):
    # find the min, max, line qty, for bins
    low = np.inf
    high = -np.inf

    loop = 0

    fh.seek(0)
    for chunk in pd.read_table(fh, header=None, chunksize=chunksize, sep='\t'):
        low = np.minimum(chunk.iloc[:, 2].min(), low) #btw, iloc is way slower than numpy array indexing
        high = np.maximum(chunk.iloc[:, 2].max(), high) #you might wanna import and do the chunks with numpy
        loop += 1
    lines = loop*chunksize

    nbins = math.ceil(math.sqrt(lines))   

    bin_edges = np.linspace(low, high, nbins + 1)
    total = np.zeros(nbins, np.int64)  # np.ndarray filled with np.uint32 zeros, CHANGED TO int64

    fh.seek(0)
    for chunk in pd.read_table(fh, header=None, chunksize=chunksize, delimiter='\t'):

        # compute bin counts over the 3rd column
        subtotal, e = np.histogram(chunk.iloc[:, 2], bins=bin_edges)  # np.ndarray filled with np.int64

        # accumulate bin counts over chunks
        total += subtotal

    ax.hist(bin_edges[:-1], bins=bin_edges, weights=total,alpha=0.5)

    return(ax,bin_edges,total)

compare = {
  'train': pd.DataFrame({'true':Y_train,'pred':y_train_pred}),
  'test': pd.DataFrame({'true':Y_test,'pred':y_test_pred})
}
gs0 = write_stringIO(compare['train'].loc[compare['train']['true']==0,'pred'])
gs1 = write_stringIO(compare['train'].loc[compare['train']['true']==1,'pred'])
chunksize = 1000

fig,ax = plt.subplots()
test_1_data = make_hist(gs0,ax)
test_2_data = make_hist(gs1,ax)
ax.set_title("ks: %f, p_in_the_v: %f" % stats.ks_2samp(test_1_data[2], test_2_data[2]))
fig.savefig('out/las_kupedes_ultramikro_v3_cv_xgb_ks.png')
Image(filename="out/las_kupedes_ultramikro_v3_cv_xgb_ks.png")

# Cluster prediction values
N_grade = 8
from sklearn.cluster import KMeans
compare = {
  'train': pd.DataFrame({'true':Y_train,'pred':y_train_pred}),
  'test': pd.DataFrame({'true':Y_test,'pred':y_test_pred})
}
CUTOFF = 0.15
#X_cluster = compare['train'].loc[compare['train']['true']==1,'pred']
X_cluster = compare['train'].loc[:,'pred']
X_cluster = np.array(X_cluster).reshape(-1,1)
X_cluster_label = compare['train'].loc[:,'true']
X_cluster_label = np.array(X_cluster_label).reshape(-1,1)
kmeans = KMeans(n_clusters=N_grade)
kmeans.fit(X=X_cluster)
Y_cluster = kmeans.predict(X=X_cluster)

gs = {}
for i in range(N_grade):
  gs[i] = X_cluster[Y_cluster==i]

for i in range(N_grade):
  gs[i] = gs[i].reshape(len(gs[i]))
  
chunksize = 500
fig,ax = plt.subplots()
bin_edges = np.linspace(X_cluster.min(), X_cluster.max(), chunksize + 1)
for i in range(N_grade):
  ax.hist(gs[i], bins=bin_edges,alpha=0.5)

#ax.set_title("ks: %f, p_in_the_v: %f" % stats.ks_2samp(test_1_data[2], test_2_data[2]))
fig.savefig('out/las_kupedes_ultramikro_v3_cv_xgb_grade.png')
Image(filename="out/las_kupedes_ultramikro_v3_cv_xgb_grade.png")

r1=[]; r2=[]; r3=[]; r4=[]; r5=[]
grade = pd.DataFrame(columns=['Score_AVG','Score_MIN','Score_MAX','Total','PD(%)'])
for n in range(N_grade):
    r1.append(X_cluster[Y_cluster==n].mean())
    r2.append(X_cluster_label[Y_cluster==n].reshape(len(X_cluster_label[Y_cluster==n])).sum() *100 / len(X_cluster_label[Y_cluster==n]))
    r3.append(len(X_cluster_label[Y_cluster==n]))
    r4.append(X_cluster[Y_cluster==n].min())
    r5.append(X_cluster[Y_cluster==n].max())

grade['Score_AVG'] = r1
grade['Score_MIN'] = (1-np.array(r5)*2)*1000
grade['Score_MAX'] = (1-np.array(r4)*2)*1000
grade['Total'] = r3
grade['PD(%)'] = r2
grade['Proba_MIN'] = r4
grade['Proba_MAX'] = r5
grade = grade.sort_index(by='PD(%)').reset_index(drop=True)
grade['Rating'] = np.arange(1,N_grade+1)
grade = grade.loc[:,['Rating','Proba_MIN','Proba_MAX','Score_MIN','Score_MAX','Total','PD(%)']]
grade.loc[N_grade-1,'Score_MIN'] = 0
grade.loc[0,'Score_MAX'] = 1000
grade['Score_MIN'] = np.round(grade['Score_MIN'])
grade['Score_MAX'] = np.round(grade['Score_MAX'])
grade
grade.to_csv("out/las_kupedes_ultramikro_v3_cv_xgb_grade.csv",index=False)

# Model evaluation: AUC & Gini
threshold = 0.12
fig,ax = plt.subplots(1,3)
fig.set_size_inches(15,5)
plot_cm(ax[0],  Y_train, y_train_pred, [0,1], 'Confusion matrix (TRAIN)', threshold)
#plot_cm(ax[0],  Y_test, y_test_pred,   [0,1], 'Confusion matrix (TEST)', threshold)
plot_cm(ax[1],  Y_test, y_test_pred,   [0,1], 'Confusion matrix (TEST)', threshold)
plot_auc(ax[2], Y_train, y_train_pred, Y_test, y_test_pred, threshold) 
#plot_auc(ax[2], Y_test, y_test_pred, Y_test, y_test_pred, threshold) 
fig.savefig('out/tmp.png')
# Display image
from IPython.display import Image
Image(filename="out/tmp.png")

# Model evaluation: AUC & Gini
GRADE_CUTOFF = 6
threshold = grade.Proba_MAX[-1+GRADE_CUTOFF]
fig,ax = plt.subplots(1,3)
fig.set_size_inches(15,5)
plot_cm(ax[0],  Y_train, y_train_pred, [0,1], 'Confusion matrix (TRAIN)', threshold)
#plot_cm(ax[0],  Y_test, y_test_pred,   [0,1], 'Confusion matrix (TEST)', threshold)
plot_cm(ax[1],  Y_test, y_test_pred,   [0,1], 'Confusion matrix (TEST)', threshold)
plot_auc(ax[2], Y_train, y_train_pred, Y_test, y_test_pred, threshold) 
#plot_auc(ax[2], Y_test, y_test_pred, Y_test, y_test_pred, threshold) 
fig.savefig('out/las_kupedes_ultramikro_v3_cv_xgb_roc_{}.png'.format(GRADE_CUTOFF))
# Display image
from IPython.display import Image
Image(filename="out/las_kupedes_ultramikro_v3_cv_xgb_roc_{}.png".format(GRADE_CUTOFF))

# Model evaluation: PD
CUTOFF = 0.0402575508
X1 = le.transform(X)
X1 = pipeline_preprocess.transform(X1)
y_proba = bst.predict(X1)
#y_pred = y_proba.map(lambda x: 1 if x >= CUTOFF else 0)
#y_proba.min()
#y_proba.max()
bins = np.arange(y_proba.min(),y_proba.max(), (y_proba.max() - y_proba.min())/10)
bins = np.append(bins,1.0)
bins = [0.01098923, 0.05086363, 0.09073803, 0.13061243, 0.17048683,1.        ]
calculate_PD(y_proba, Y, bins=bins)
calculate_PD(y_proba, Y, N=10)

# XGBoost explainer
bst1 = bst
bst = bst1.model[0]
tree_lst = model2table(bst)
d_train = xgb.DMatrix(X_train,feature_names=colname)
leaf_lst = bst.predict(d_train, pred_leaf=True)
bst_logit = logit_contribution(tree_lst, leaf_lst)

# Feature Importances
feature_rank = bst.get_importances()

CUTOFF = 0.15
X_woe = le.transform(X)
X_woe = pipeline_preprocess.transform(X_woe)
y_proba = bst.predict(X_woe)
y_pred = y_proba.map(lambda x: 1 if x >= CUTOFF else 0)

calculate_woe_numerical(X['jangka_waktu'], y_pred, bins=[0,12,18,24]) 
#1.00175067
calculate_woe_numerical(X['usia'], y_pred, bins=[16, 22, 27, 35, 45, 55, 65, 90]) 
# 0.34936246628147705
calculate_woe_numerical(X['plafond'], y_pred, bins=[4500000,10000000, 15000000, 20000000, 25000000])
# 0.0473384
calculate_woe_numerical(X['simpanan'], y_pred, bins=[-0.1, 100000, 250000, 500000, 1000000, 5000000, 10000000, 50000000]) 
# 0.1352424
calculate_woe_numerical(X['lama_bekerja'], y_pred, bins=[-1,0, 2, 6, 10, 20, 50]) 
# 0.41425081
calculate_woe_numerical(X['ratio_pendatapan_to_angsuran'], y_pred, bins=[0.95, 1.33, 2.00, 3.00, 10.00]) 
# 0.045912297
calculate_woe_categorical(X['jenis_kelamin'], y_pred) 
# 0.0974077
calculate_woe_categorical(X['tujuan_penggunaan'], y_pred) 
# 0.04064867

import numpy as np
import math
from scipy import stats
from sklearn.utils.multiclass import type_of_target
class WOE:
    def __init__(self):
        self._WOE_MIN = -20
        self._WOE_MAX = 20

    def woe(self, X, y, event=1):
        '''
        Calculate woe of each feature category and information value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable which should be binary
        :param event: value of binary stands for the event to predict
        :return: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
                 numpy array of information value of each feature
        '''
        self.check_target_binary(y)
        X1 = self.feature_discretion(X)

        res_woe = []
        res_iv = []
        for i in range(0, X1.shape[-1]):
            x = X1[:, i]
            woe_dict, iv1 = self.woe_single_x(x, y, event)
            res_woe.append(woe_dict)
            res_iv.append(iv1)
        return np.array(res_woe), np.array(res_iv)

    def woe_single_x(self, x, y, event=1):
        '''
        calculate woe and information for a single feature
        :param x: 1-D numpy starnds for single feature
        :param y: 1-D numpy array target variable
        :param event: value of binary stands for the event to predict
        :return: dictionary contains woe values for categories of this feature
                 information value of this feature
        '''
        self.check_target_binary(y)

        event_total, non_event_total = self.count_binary(y, event=event)
        x_labels = np.unique(x)
        woe_dict = {}
        iv = 0
        for x1 in x_labels:
            y1 = y[np.where(x == x1)[0]]
            event_count, non_event_count = self.count_binary(y1, event=event)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            if rate_event == 0:
                woe1 = self._WOE_MIN
            elif rate_non_event == 0:
                woe1 = self._WOE_MAX
            else:
                woe1 = math.log(rate_event / rate_non_event)
            woe_dict[x1] = woe1
            iv += (rate_event - rate_non_event) * woe1
        return woe_dict, iv

    def woe_replace(self, X, woe_arr):
        '''
        replace the explanatory feature categories with its woe value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param woe_arr: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
        :return: the new numpy array in which woe values filled
        '''
        if X.shape[-1] != woe_arr.shape[-1]:
            raise ValueError('WOE dict array length must be equal with features length')

        res = np.copy(X).astype(float)
        idx = 0
        for woe_dict in woe_arr:
            for k in woe_dict.keys():
                woe = woe_dict[k]
                res[:, idx][np.where(res[:, idx] == k)[0]] = woe * 1.0
            idx += 1

        return res

    def combined_iv(self, X, y, masks, event=1):
        '''
        calcute the information vlaue of combination features
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable
        :param masks: 1-D numpy array of masks stands for which features are included in combination,
                      e.g. np.array([0,0,1,1,1,0,0,0,0,0,1]), the length should be same as features length
        :param event: value of binary stands for the event to predict
        :return: woe dictionary and information value of combined features
        '''
        if masks.shape[-1] != X.shape[-1]:
            raise ValueError('Masks array length must be equal with features length')

        x = X[:, np.where(masks == 1)[0]]
        tmp = []
        for i in range(x.shape[0]):
            tmp.append(self.combine(x[i, :]))

        dumy = np.array(tmp)
        # dumy_labels = np.unique(dumy)
        woe, iv = self.woe_single_x(dumy, y, event)
        return woe, iv

    def combine(self, list):
        res = ''
        for item in list:
            res += str(item)
        return res

    def count_binary(self, a, event=1):
        event_count = (a == event).sum()
        non_event_count = a.shape[-1] - event_count
        return event_count, non_event_count

    def check_target_binary(self, y):
        '''
        check if the target variable is binary, raise error if not.
        :param y:
        :return:
        '''
        y_type = type_of_target(y)
        if y_type not in ['binary']:
            raise ValueError('Label type must be binary')

    def feature_discretion(self, X):
        '''
        Discrete the continuous features of input data X, and keep other features unchanged.
        :param X : numpy array
        :return: the numpy array in which all continuous features are discreted
        '''
        temp = []
        for i in range(0, X.shape[-1]):
            x = X[:, i]
            x_type = type_of_target(x)
            if x_type == 'continuous':
                x1 = self.discrete(x)
                temp.append(x1)
            else:
                temp.append(x)
        return np.array(temp).T

    def discrete(self, x):
        '''
        Discrete the input 1-D numpy array using 5 equal percentiles
        :param x: 1-D numpy array
        :return: discreted 1-D numpy array
        '''
        res = np.array([0] * x.shape[-1], dtype=int)
        for i in range(5):
            point1 = stats.scoreatpercentile(x, i * 20)
            point2 = stats.scoreatpercentile(x, (i + 1) * 20)
            x1 = x[np.where((x >= point1) & (x <= point2))]
            mask = np.in1d(x, x1)
            res[mask] = (i + 1)
        return res

    @property
    def WOE_MIN(self):
        return self._WOE_MIN
    @WOE_MIN.setter
    def WOE_MIN(self, woe_min):
        self._WOE_MIN = woe_min
    @property
    def WOE_MAX(self):
        return self._WOE_MAX
    @WOE_MAX.setter
    def WOE_MAX(self, woe_max):
        self._WOE_MAX = woe_max



# Save model
#predict_fn_xgb = lambda x: xgbst.predict_proba(x)
joblib.dump(bst, './model/las_kupedes_ultramikro_v3_cv_xgb.sav')
joblib.dump(bst1, './model/las_kupedes_ultramikro_v3_cv_xgb.sav')
# Load model
bst = joblib.load('./model/las_kupedes_ultramikro_v3_cv_xgb.sav')


  


"""
### CV - RandomForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
# Set the parameters by cross-validation
tuned_parameters = [{
    'n_estimators': [10],
    'min_samples_leaf' : [5,10],
    'class_weight' : [{0:1,1:10}]
}]
score_target = 'roc_auc'
#for score in scores:
#print("# Tuning hyper-parameters for %s" % score)

# Create RF CV model
clf = GridSearchCV(
  RandomForestClassifier(), 
  tuned_parameters, cv=5,
  # scoring='%s_macro' % score
)
# Train the model using the training sets
N = X_train.shape[0]
listidx = random.sample(range(X_train.shape[0]),  N)
# Perform training
#clf.fit(X_train, Y_train)
clf.fit(X_train_upsampled[listidx,:], Y_train[listidx])

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()

y_true, y_pred = Y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
# Predict testing data
y_train_pred = clf.predict_proba(X_train)[:,1]
y_test_pred = clf.predict_proba(X_test)[:,1]
test_compare["pred_cv_RF"] = y_test_pred

# Model evaluation: prediction distribution
feature = "pred_cv_RF"
N = test_compare.shape[0]
listidx = random.sample(test_compare.index.values.tolist(),  N)
disp = test_compare.loc[listidx,:]
sns.distplot(a=disp.loc[disp.true==0,feature], hist=True)
sns.distplot(a=disp.loc[disp.true==1,feature], hist=True)

plot_ks(y_test_pred, Y_test)

# Model evaluation: AUC
threshold = 0.35
fig,ax = plt.subplots(1,3)
fig.set_size_inches(15,5)
plot_cm(ax[0],  Y_train, y_train_pred, [0,1], 'Confusion matrix (TRAIN)', threshold)
plot_cm(ax[1],  Y_test, y_test_pred,   [0,1], 'Confusion matrix (TEST)', threshold)
plot_auc(ax[2], Y_train, y_train_pred, Y_test, y_test_pred, threshold) 
fig.savefig('out/las_kupedes_v3_cv_rf.png')
Image(filename="out/las_kupedes_v3_cv_rf.png")

# Save model
joblib.dump(clf, './model/las_kupedes_v3_cv_rf.sav')
# Load model
#clf = joblib.load('./model/las_kupedes_v3_cv_rf.sav')
"""
  
  
