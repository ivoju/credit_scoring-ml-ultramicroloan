import pandas as pd

from sklearn.externals import joblib

from common import *
from common.ml_dev import *
from common.ml_prod import *
#import xgboost_explainer as xgb_exp

#support for reading excel files
#import xlrd

#%matplotlib inline
pd.options.display.html.table_schema = True
pd.options.display.max_columns = 99
#%env JOBLIB_TEMP_FOLDER=/tmp

class result(object):
    def __init__(self, pd, score, rating):
        self.pd = pd
        self.score = score
        self.rating = rating

def getscore_ultramicroloan(df_create, df_grade):
    # ## -- Create data frame that equal for input model
    # data = {'tgl_mulai': ['1999-01-01'], 'jangka_waktu': [24], 'plafond': [25000000.0], 'jenis_kelamin': ['M'], 
    #         'pendidikan': ['SD/SMP/SMU/SMK'], 'status_nikah': ['BelumMenikah'], 'jumlah_anak_dlm_tanggungan': [1], 
    #         'kepemilikan_tempat_tinggal': ['MilikSendiri'], 'debitur_lama': [1.0], 'tujuan_penggunaan': ['KI'], 
    #         'status_bukti_kepemilikan_agunan': ['XXX'], 'lama_bekerja': [1], 'pemasaran': [0.0], 'simpanan': [1238400.0], 
    #         'usia': [71.0], 'ratio_pendatapan_to_angsuran': [1.41481], 'punya_usaha_sampingan': [1.0], 'ratio_likuidasi_agunan_to_plafond': [0.0], 'target': [0]}
    # df_create = pd.DataFrame(data=d)
    # df_create = df_create[['tgl_mulai', 'jangka_waktu', 'plafond', 'jenis_kelamin', 'pendidikan', 'status_nikah', 
    #                         'jumlah_anak_dlm_tanggungan', 'kepemilikan_tempat_tinggal', 'debitur_lama', 'tujuan_penggunaan', 
    #                         'status_bukti_kepemilikan_agunan', 'lama_bekerja', 'pemasaran', 'simpanan', 'usia', 'ratio_pendatapan_to_angsuran', 
    #                         'punya_usaha_sampingan', 'ratio_likuidasi_agunan_to_plafond', 'target']]

    ## -- Drop unimportant columns (same process as build model step)
    df_create = df_create.drop(labels=['status_bukti_kepemilikan_agunan','ratio_likuidasi_agunan_to_plafond','pemasaran'], axis=1)
    dtype = df_create.dtypes.drop([
      'tgl_mulai','target',
    #  'jenis_fasilitas','bln_mulai',
    #  'suku_bunga','baru_perpanjangan','jabatan','bidang_pekerjaan','jenis_pekerjaan','region'
    ])
    list_col = dtype.index.values.tolist()

    ## -- Just select importany columns (same process as build model step)
    X = df_create.loc[:, list_col]
    X.reset_index(drop=True, inplace=True)

    ## -- Load pipeline preprocess variable (including one hot encoding variable)
    dtype,categorical_feat_classes,list_col_cat,list_idx_cat,categorical_onehot_idx,categorical_onehot_nval,colname = joblib.load('./model/las_kupedes_ultramikro_v3_var.sav')
    #le,pipeline_preprocess = joblib.load('./model/las_kupedes_ultramikro_v3_preprocess.sav')
    le,pipeline_preprocess = joblib.load('./model/las_kupedes_ultramikro_v3_preprocess_wo_scaler.sav')

    ## -- Transfor input model variable so its have suitable format for modeling (array of one hot encoding format)
    X = le.transform(X)
    X = pipeline_preprocess.transform(X)

    ## -- Load model and predict
    bst = joblib.load('./model/las_kupedes_ultramikro_v3_cv_xgb.sav')
    Y = bst.predict(X)
    Y = Y.values 

    ## -- Read rule of credit scoring grade and transform prediction result to grade level
    # df_grade = pd.read_csv("./model/las_kupedes_ultramikro_v3_cv_xgb_grade.csv")

    if Y>=0 and Y<=df_grade.loc[0,'Proba_MAX']:
        score = (((((Y-df_grade.loc[0,'Proba_MIN'])*(df_grade.loc[0,'Score_MIN']-df_grade.loc[0,'Score_MAX']))/(df_grade.loc[0,'Proba_MAX']-df_grade.loc[0,'Proba_MIN'])))+df_grade.loc[0,'Score_MAX'])
        rating = df_grade.loc[0,'Rating'] 
    elif Y>=df_grade.loc[len(df_grade.index)-2,'Proba_MAX'] and Y<=1:
            index = len(df_grade.index)-1
            score = (((((Y-df_grade.loc[index-1,'Proba_MAX'])*(df_grade.loc[index,'Score_MIN']-df_grade.loc[index,'Score_MAX']))/(1-df_grade.loc[index-1,'Proba_MAX'])))+df_grade.loc[index,'Score_MAX'])
            rating = df_grade.loc[index,'Rating']  
    else:    
        for index in range (1,(len(df_grade.index)-1)):
            if Y>=df_grade.loc[index-1,'Proba_MAX'] and Y<=df_grade.loc[index,'Proba_MAX']:
                score = (((((Y-df_grade.loc[index-1,'Proba_MAX'])*(df_grade.loc[index,'Score_MIN']-df_grade.loc[index,'Score_MAX']))/(df_grade.loc[index,'Proba_MAX']-df_grade.loc[index-1,'Proba_MAX'])))+df_grade.loc[index,'Score_MAX'])
                rating = df_grade.loc[index,'Rating']
                break

    pd = Y[0]
    score = score[0]

    resp = result(pd, score, rating)

    return resp
