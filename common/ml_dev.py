from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import itertools
from scipy import stats

def balanceDataset(dataset, label): 

    #Re-balancing (weighting) of records to be used in the logistic loss objective function
    numNegatives = dataset.filter("{} == 0".format(label)).count()
    datasetSize = dataset.count()
    balancingRatio = (datasetSize - numNegatives) * 1.0 / datasetSize

    calculateWeights = F.udf(
      lambda d: 1 * balancingRatio if d==0.0 else (1 * (1.0 - balancingRatio)),
      DoubleType()
    )

    weightedDataset = dataset.withColumn("classWeightCol", calculateWeights(F.col(label)))
    return weightedDataset

def plot_cm(ax, y_true, y_pred, classes, title, th=0.5, cmap=plt.cm.Blues):
    y_pred_labels = (y_pred>th).astype(int)
    
    cm = confusion_matrix(y_true, y_pred_labels)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

def plot_auc(ax, y_train, y_train_pred, y_test, y_test_pred, th=0.5):

    y_train_pred_labels = (y_train_pred>th).astype(int)
    y_test_pred_labels  = (y_test_pred>th).astype(int)

    fpr_train, tpr_train, _ = roc_curve(y_train,y_train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)
    acc_train = accuracy_score(y_train, y_train_pred_labels)

    fpr_test, tpr_test, _ = roc_curve(y_test,y_test_pred)
    roc_auc_test = auc(fpr_test, tpr_test)
    acc_test = accuracy_score(y_test, y_test_pred_labels)

    ax.plot(fpr_train, tpr_train)
    ax.plot(fpr_test, tpr_test)

    ax.plot([0, 1], [0, 1], 'k--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    
    train_text = 'train acc = {:.3f}, auc = {:.2f}'.format(acc_train, roc_auc_train)
    test_text = 'test acc = {:.3f}, auc = {:.2f}'.format(acc_test, roc_auc_test)
    ax.legend([train_text, test_text])

#def plot_ks(y_pred, y_true):
#    result = pd.DataFrame({
#        "predicted":y_pred,
#        "bad":y_true.astype(int),
#        "good":np.abs(1-y_true).astype(int)
#    })
#    result = result.sort_values(by='predicted', ascending=False)
#    result['bucket'] = pd.qcut(result.predicted, q=10, labels=np.arange(10)+1)
#    
#    grouped = result.groupby('bucket',as_index=False)
#    
#    agg = pd.DataFrame({
#        "min_scr":grouped.min().predicted, 
#        "max_scr":grouped.max().predicted,
#        "tot_bad": grouped.sum().bad,
#        "tot_good":grouped.count().good - grouped.sum().bad,
#        "total":grouped.count().bad
#    })
#
#    # agg = agg.sort_index(by='min_scr').reset_index(drop=True)
#    agg = agg[['min_scr']+[x for x in agg.columns.values if x!='min_scr']]
#
#    agg['pct_bad'] = (agg.tot_bad / result.bad.sum())#.apply('{0:.2%}'.format)
#    agg['pct_good'] = (agg.tot_good / result.good.sum())#.apply('{0:.2%}'.format)
#    agg['odds'] = (agg.tot_good / agg.tot_bad).apply('{0:.2f}'.format)
#
#    agg['cum_bad'] = agg.pct_bad.cumsum()
#    agg['cum_good'] = agg.pct_good.cumsum()
#    agg['ks'] = np.abs(np.round(agg.cum_bad - agg.cum_good, 4))
#    # agg['pct_bad']  = (agg.tot_bad / agg.total).apply('{0:.2%}'.format)
#    # agg['ks'] = np.abs(np.round(((agg.tot_bad / result.bad.sum()).cumsum() - (agg.tot_good / result.good.sum()).cumsum()), 4))
#    agg['max_ks'] = agg.ks.apply(lambda x: '<----' if x == agg.ks.max() else '')
#    
#    fig,ax = plt.subplots(1,1)
#    fig.set_size_inches(15,5)
#    ax.plot(agg.index.values, agg.cum_bad)
#    ax.plot(agg.index.values, agg.cum_good)
#    
#    return agg

def plot_ks(y_pred, y_true, N=10, bins=None):
    result = pd.DataFrame({
        "predicted":y_pred,
        "bad":y_true.astype(int),
        "good":np.abs(1-y_true).astype(int)
    })
    result = result.sort_values(by='predicted', ascending=False)
    if bins!=None:
      result['bucket'] = pd.cut(y_pred, bins=bins)
    else:
      result['bucket'] = pd.qcut(result.predicted, q=N, labels=np.arange(N)+1)
    
    grouped = result.groupby('bucket',as_index=False)
    
    agg = pd.DataFrame({
        "min_scr":grouped.min().predicted, 
        "max_scr":grouped.max().predicted,
        "tot_bad": grouped.sum().bad,
        "tot_good":grouped.count().good - grouped.sum().bad,
        "total":grouped.count().bad
    })

    # agg = agg.sort_index(by='min_scr').reset_index(drop=True)
    agg = agg[['min_scr']+[x for x in agg.columns.values if x!='min_scr']]

    agg['pct_bad'] = (agg.tot_bad / result.bad.sum())#.apply('{0:.2%}'.format)
    agg['pct_good'] = (agg.tot_good / result.good.sum())#.apply('{0:.2%}'.format)
    agg['odds'] = (agg.tot_good / agg.tot_bad).apply('{0:.2f}'.format)

    agg['cum_bad'] = agg.pct_bad.cumsum()
    agg['cum_good'] = agg.pct_good.cumsum()
    agg['ks'] = np.abs(np.round(agg.cum_bad - agg.cum_good, 4))
    # agg['pct_bad']  = (agg.tot_bad / agg.total).apply('{0:.2%}'.format)
    # agg['ks'] = np.abs(np.round(((agg.tot_bad / result.bad.sum()).cumsum() - (agg.tot_good / result.good.sum()).cumsum()), 4))
    agg['max_ks'] = agg.ks.apply(lambda x: '<----' if x == agg.ks.max() else '')
    
    fig,ax = plt.subplots(1,1)
    fig.set_size_inches(15,5)
    ax.plot(agg.index.values, agg.cum_bad)
    ax.plot(agg.index.values, agg.cum_good)
    
    return agg
  
def calculate_PD(y_pred, y_true, N=10, bins=None):
    
#    test_compare = pd.DataFrame({"true":y_true})
#    test_compare["pred"] = y_pred
#    
##    disp = test_compare.loc[test_compare.true==1,:]
#    disp = test_compare
#    
#    y_pr = disp['pred']
##    y_pd = stats.norm.cdf(y_pr     , y_pr.mean(), y_pr.std())
#    y_tr = disp['true']
    
    result = pd.DataFrame({
        "predicted":y_pred,
        "bad":y_true.astype(int),
        "good":np.abs(1-y_true).astype(int)
    })
    result = result.sort_values(by='predicted', ascending=False)
#    result['bucket'] = pd.qcut(result.predicted, q=N, labels=np.arange(N)+1)

#    bins=[0]
#    RANGE = np.arange(0.1,1.1,0.1)
#    for i in np.arange(0.1,1.1,0.1):
#      bins.append(stats.norm.ppf(i, loc=y_pr.mean(), scale=y_pr.std()))
#    bins = np.arange(0,1.1,0.1)

    if type(bins) in [list,np.ndarray]:
      result['bucket'] = pd.cut(result.predicted, bins=bins)
    else:
      result['bucket'] = pd.qcut(result.predicted, q=N, labels=np.arange(N)+1)
    
    grouped = result.groupby('bucket',as_index=False)
    
    agg = pd.DataFrame({
        "PD_min":grouped.min().predicted, 
        "PD_max":grouped.max().predicted, 
#        "PD_avg":grouped.mean().predicted,
#        "PD_min":grouped.min().PD, 
#        "PD_max":grouped.max().PD, 
#        "PD_avg":grouped.mean().PD,
        "tot_bad": grouped.sum().bad,
        "tot_good":grouped.count().good - grouped.sum().bad,
        "total":grouped.count().bad
    })
#    agg['pct_bad'] = agg['tot_bad'] / agg['total']
#    agg['pct_good'] = agg['tot_good'] / agg['total']
    agg['pct_bad'] = (agg.tot_bad / result.bad.sum())#.apply('{0:.2%}'.format)
    agg['pct_good'] = (agg.tot_good / result.good.sum())#.apply('{0:.2%}'.format)
    agg = agg.loc[:,['PD_min','PD_max','tot_bad','total','pct_bad']]

    # agg = agg.sort_index(by='min_scr').reset_index(drop=True)
#    agg = agg[['PD_min']+[x for x in agg.columns.values if x!='min_scr']]

    return agg
  
def calculate_woe_numerical(y_pred, y_true, N=10, bins=None):
    result = pd.DataFrame({
        "feature":y_pred,
        "bad":y_true.astype(int),
        "good":np.abs(1-y_true).astype(int)
    })
    result = result.sort_values(by='feature', ascending=False)
    if bins!=None:
      result['bucket'] = pd.cut(y_pred, bins=bins,right=True)
    else:
      result['bucket'] = pd.qcut(result.feature, q=N, labels=np.arange(N)+1)
    
    grouped = result.groupby('bucket',as_index=True)
    
    agg = pd.DataFrame({
        "min_val":grouped.min().feature, 
        "max_val":grouped.max().feature,
        "tot_bad": grouped.sum().bad,
        "tot_good":grouped.count().good - grouped.sum().bad,
        "total":grouped.count().bad
    })
    
    agg = agg[['min_val']+[x for x in agg.columns.values if x!='min_val']]
    agg['pct_bad'] = (agg.tot_bad / result.bad.sum())#.apply('{0:.2%}'.format)
    agg['pct_good'] = (agg.tot_good / result.good.sum())#.apply('{0:.2%}'.format)
    agg['distr_bad'] = (agg.tot_bad / result.bad.sum())#.apply('{0:.2%}'.format)
    agg['distr_good'] = (agg.tot_good / result.good.sum())#.apply('{0:.2%}'.format)
    agg['distr_margin'] = agg.distr_bad - agg.distr_good
    agg['WoE'] = np.log(agg.distr_bad / agg.distr_good)
    agg['IV'] = agg.distr_margin * agg['WoE']
    
    iv = agg.IV.sum()
    return iv,agg
  
def calculate_woe_categorical(y_pred, y_true):
    result = pd.DataFrame({
        "feature":y_pred,
        "bad":y_true.astype(int),
        "good":np.abs(1-y_true).astype(int)
    })
    result = result.sort_values(by='feature', ascending=False)
    
    grouped = result.groupby('feature',as_index=True)
    
    agg = pd.DataFrame({
        "tot_bad": grouped.sum().bad,
        "tot_good":grouped.count().good - grouped.sum().bad,
        "total":grouped.count().bad
    })
    agg['pct_bad'] = (agg.tot_bad / result.bad.sum())#.apply('{0:.2%}'.format)
    agg['pct_good'] = (agg.tot_good / result.good.sum())#.apply('{0:.2%}'.format)
    agg['distr_bad'] = (agg.tot_bad / result.bad.sum())#.apply('{0:.2%}'.format)
    agg['distr_good'] = (agg.tot_good / result.good.sum())#.apply('{0:.2%}'.format)
    agg['distr_margin'] = agg.distr_bad - agg.distr_good
    agg['WoE'] = np.log(agg.distr_bad / agg.distr_good)
    agg['IV'] = agg.distr_margin * agg['WoE']
    
    iv = agg.IV.sum()
    return iv,agg
  
  