IN_dataset = "./data/las_dataset_kupedes_clean_v4.csv"
IN_label = "kolektibilitas"

## ================================================================================================== ##
### Pull data from HDFS

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.rdd import reduce
from pyspark.sql.utils import AnalysisException, ParseException
import logging
import re
from datetime import datetime
import pandas as pd
pd.options.display.html.table_schema = True
pd.options.display.max_rows = 999

#create spark session
spark = SparkSession \
    .builder \
    .appName("LAS Kupedes v3 Model") \
    .config("spark.dynamicAllocation.enabled", "false") \
    .config("spark.executor.instances", "1") \
    .config("spark.executor.cores", "2") \
    .config("spark`.executor.memory", "3g") \
    .config("spark.network.timeout", 60) \
    .config("spark.yarn.executor.memoryOverhead", "864M") \
    .config('spark.kryoserializer.buffer.max value','30582400')\
    .config('spark.kryoserializer.buffer.max value.mb','30582400')\
    .enableHiveSupport() \
    .getOrCreate()
#spark.conf.set('spark.kryoserializer.buffer.max value','1g')
#spark.sparkContext.getConf().getAll()

df = spark.read.table("datasci.creditscoring_las_kupedes_clean_v4")
#df.limit(100).toPandas()
df = df.toPandas()
df.to_csv(IN_dataset, index=None, header=True)
df = pd.read_csv(IN_dataset)
## ================================================================================================== ##

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

df = pd.read_csv(IN_dataset)
#df = pd.read_csv(IN_dataset, nrows=30000)

# Data Preparation
## Feature Engineering #0

data = df.copy()

data = data.drop(axis=1, labels=[
    "no_rekening",
    "cif",
    "id_aplikasi",
    "id_kredit",
    "fid_cif_las",
    "tp_produk",
])

data['debitur_baru'] = data.debitur_baru.map(lambda x: 0 if x==True else (1 if x==False else None))
data = data.rename(columns={"debitur_baru":"debitur_lama"})

def convertStatusNikah(x):
    if x==1:
        return "Menikah"
    elif x==2:
        return "BelumMenikah"
    elif x==3:
        return "JandaDuda"
    else:
        return None
data['status_nikah'] = data.status_nikah.map(lambda x: convertStatusNikah(x))

def convertDateDiff(df, label1, label2):
    x1 = df[label1]
    x2 = df[label2]
    try:
        x = x2 - x1
        x = round(x.days / 364)
    except ValueError:
        x = None
    return x

data['tgl_mulai'] = pd.to_datetime(data.tgl_mulai)
data['tanggal_lahir'] = pd.to_datetime(data.tanggal_lahir, format="%d-%m-%Y")
#data['tanggal_mulai_bekerja'] = pd.to_datetime(data.tanggal_mulai_bekerja, format="%Y-%m-%d")
data['tanggal_mulai_bekerja'] = pd.to_datetime(data.tanggal_mulai_bekerja, format="%d-%m-%Y")

data['usia'] = data.apply(lambda x: convertDateDiff(x, 'tanggal_lahir', 'tgl_mulai'), axis=1)
data['lama_bekerja'] = data.apply(lambda x: convertDateDiff(x, 'tanggal_mulai_bekerja', 'tgl_mulai'), axis=1)
data['lama_bekerja'] = data.lama_bekerja.map(lambda x: max(x,0))

data = data.drop(axis=1, labels=["tanggal_lahir","tanggal_mulai_bekerja"])

def convertAgunanStatusKepemilikan(x):
    try:
        if 'Kwitansi' in x:
            return 'Kwitansi'
        elif (('Bukti Kepemilikan Kendaraan Bermotor' in x) or ('BPKB' in x)):
            return 'BPKB'
        elif (('Sertifikat Hak Milik' in x) or ('SHM' in x)):
            return 'SHM'
        elif (('Sertifikat Hak Guna Bangunan' in x) or ('SHGB' in x)):
            return 'SHGB'
        elif (('Petok' in x) or ('Letter' in x) or ('Surat Tanah di luar ' in x)):
            return 'SuratTanahDiLuarSertifikat'
        else:
            return None
    except (TypeError):
        return None
data['status_bukti_kepemilikan_agunan'] = data.status_bukti_kepemilikan_agunan.map(lambda x: convertAgunanStatusKepemilikan(x))
# data = pd.get_dummies(columns=["status_bukti_kepemilikan_agunan"], data=data)

def convertJenisKelamin(x):
    try:
        if ((x=='M') or (x=='L')):
            return 'M'
        if ((x=='F') or (x=='W')):
            return 'F'
        else:
            return None
    except (TypeError):
        return None
data['jenis_kelamin'] = data.jenis_kelamin.map(lambda x: convertJenisKelamin(x))

def convertKepemilikanTempatTinggal(x):
    if x==0:
        return "MilikSendiri"
    elif x==1:
        return "RumahDinas"
    elif x==3:
        return "SewaKontrak"
    else:
        return None
data['kepemilikan_tempat_tinggal'] = data.kepemilikan_tempat_tinggal.map(lambda x: convertKepemilikanTempatTinggal(x))

def convertTujuanPenggunaan(x):
    try:
        if x==0:
            return "KMK"
        elif x==1:
            return "PMK"
        elif x==2:
            return "KI"
        else:
            return None
    except:
        return None
data['tujuan_penggunaan'] = data.tujuan_penggunaan.map(lambda x: convertTujuanPenggunaan(x))

data['pendidikan'] = data.pendidikan.map(lambda x: 
   'TidakSekolah' if type(x) != str else (
   'TidakSekolah' if x.lstrip().rstrip().upper() == 'TIDAK SEKOLAH' else (
   'SD/SMP/SMU/SMK' if x.lstrip().rstrip().upper() in ['SD','SMP','SMA','SMU/SMK'] else (
   'Akademi/Diploma' if x.strip().upper() in ['D3','D2','DIPLOMA','AKADEMI'] else (
   'Akademi/Diploma' if 'D3' in x.strip().upper() else (
   'Akademi/Diploma' if 'DIII' in x.strip().upper() else (
   'Akademi/Diploma' if 'D2' in x.strip().upper() else (
   'Akademi/Diploma' if 'DII' in x.strip().upper() else (
   'Akademi/Diploma' if 'D1' in x.strip().upper() else (
   'Akademi/Diploma' if 'DI' in x.strip().upper() else (
   'Akademi/Diploma' if 'DIPLOMA' in x.strip().upper() else (
   'S1/S2/S3' if x.lstrip().rstrip() in ['Sarjana','Master','Doktoral'] else (
   'TIDAK SEKOLAH' ))))))))))))
)

data = data.dropna(axis=0, subset=['status_nikah','pendidikan','jenis_kelamin','usia'])

data = data.dropna(axis=0, subset=['angsuran','plafond'])

data = data.drop(['domisili'], axis=1)
data = data.dropna(axis=0, subset=['kepemilikan_tempat_tinggal'])

def jumlah_anak_dlm_tanggungan(x):
    try:
        if x>10:
            return 10
        else:
            return x
    except (TypeError):
        return None

data['jumlah_anak_dlm_tanggungan'] = data.jumlah_anak_dlm_tanggungan.map(lambda x: jumlah_anak_dlm_tanggungan(x))

data['jenis_pekerjaan'] = data['jenis_pekerjaan'].map(lambda x: None if type(x)!=str else x.rstrip())

#data = data.drop(axis=1, labels=['tgl_akad_awal','bln_akad_awal','tgl_akhir'])

data['target'] = data[IN_label].astype('category')
data = data.drop([IN_label],axis=1)

# create a 'target' column for our own convenience
print("Target variable:       '{}' -> '{}'".format('default', 'target'))

data["target"] = data['target'].map(lambda x: 1 if x>2 else (0 if x<=2 else None))

## Change the columns order
cols = list(data.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('target')) #Remove col from list
data = data[cols+['target']] #Create new dataframe with columns in the order you want

## Change the columns order
cols = list(data.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('tgl_mulai')) #Remove col from list
data = data[['tgl_mulai']+cols] #Create new dataframe with columns in the order you want

df = data.copy()
print(df.count())

## Null value imputation #0
data = df.copy()

data = data.drop(axis=1, labels=['lama_usaha'])

def fixUsahaSampingan(x):
  y = \
    x['usaha_sampingan'] if x['usaha_sampingan']>0 else (
    1 if x['pendapatan_netto_sampingan']>0 else 
    0
  )
  return y
data['usaha_sampingan'] = data.apply(lambda x: fixUsahaSampingan(x), axis=1)

data['pemasaran'] = data['pemasaran'].fillna(value=data['pemasaran'].mean())
data['simpanan'] = data['simpanan'].fillna(value=0)
data['nilai_likuidasi'] = data['nilai_likuidasi'].fillna(value=0)
data['status_bukti_kepemilikan_agunan'] = data['status_bukti_kepemilikan_agunan'].fillna(value='Lainlain')

#data = data.dropna(axis=0, subset=['jenis_pekerjaan']).reset_index(drop=True)

tmp = data.apply(lambda x: x.count(), axis=0)
tmp = [tmp.index[i] for i in range(len(tmp)) if tmp[i]<data.shape[0]]

for col in data.columns.values:
    if not(col in ['tgl_mulai','target']):
        #print(col)
        ds = data[col]
        typ = ds.dtype
        if typ==object:
            ds = ds.fillna(ds.mode()[0])
        elif type(typ)==pd.core.dtypes.dtypes.CategoricalDtype:
            ds = ds.fillna(ds.mode()[0])
        else:
            ds = ds.fillna(ds.median())
        data[col] = ds
        
print(data.count())

df = data.copy()

df.to_csv(index=None, path_or_buf="./data/las_dataset_kupedes_clean_v3_eda_v0.csv")

### Feature engineering #1
df = pd.read_csv("./data/las_dataset_kupedes_clean_v3_eda_v0.csv")
data = df.copy()

data["ratio_pendatapan_to_angsuran"] = (data["pendapatan_netto_utama"]+data["pendapatan_netto_sampingan"]) / data["angsuran"]
data['ratio_pendatapan_to_angsuran'] = data.ratio_pendatapan_to_angsuran.map(lambda x: 
        10 if (x > 10) else x
    )
data = data.drop(["pendapatan_netto_utama","angsuran"], axis=1)
data["pendapatan_netto_sampingan"] = data.pendapatan_netto_sampingan.map(lambda x: 1 if x>0 else 0)
data['punya_usaha_sampingan'] = data.apply(lambda x: max(x['pendapatan_netto_sampingan'],x['usaha_sampingan']), axis=1)
data = data.drop(["pendapatan_netto_sampingan","usaha_sampingan"], axis=1)
data["ratio_likuidasi_agunan_to_plafond"] = data["nilai_likuidasi"] / data["plafond"]
data = data.drop(["nilai_likuidasi"], axis=1)

## Change the columns order
cols = list(data.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('target')) #Remove col from list
data = data[cols+['target']] #Create new dataframe with columns in the order you want

df = data.copy()

df.to_csv(index=None, path_or_buf="./data/las_dataset_kupedes_clean_v3_eda_v1.csv")


### Exploratory data analysis #1
df = pd.read_csv("./data/las_dataset_kupedes_clean_v3_eda_v1.csv")
print(df.count())

#### Jangka waktu
feature = "status_bukti_kepemilikan_agunan"
data = df.loc[:,:]
df_cnt = (data.groupby([feature])['target']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('percentage')
         )
df_cnt = df_cnt.loc[df_cnt.target==1,:]
ax = sns.barplot(x=feature, y="percentage", hue="target", data=df_cnt)
#ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
df_cnt.sort_values(by="percentage")

#### Ratio_pendatapan_utama_to_angsuran
feature = "ratio_pendatapan_to_angsuran"
disp = df.loc[df[feature]>=0,:]
disp = disp.loc[disp[feature]<10,:]
N = disp.shape[0]
listidx = random.sample(disp.index.values.tolist(),  N)
disp = df.loc[listidx,:]
sns.distplot(a=disp.loc[disp.target==0,feature], hist=True)
sns.distplot(a=disp.loc[disp.target==1,feature], hist=True)

### Feature engineering #2
df = pd.read_csv("./data/las_dataset_kupedes_clean_v3_eda_v1.csv")
print(df.count())

### Add failed loan virtual data
#df_add = pd.read_csv("./data/las_dataset_kupedes_sample_fake_npl.csv")
#df_add = df_add.loc[:,df.columns.values]
#df = pd.concat([df,df_add])


df = df.drop([
  'bln_mulai',
  'suku_bunga',
  'bidang_pekerjaan','jenis_pekerjaan','jabatan','region',
  'baru_perpanjangan','jenis_fasilitas',
], axis=1)

df = df.loc[df['usia'] >= 16,:]
df = df.loc[df['ratio_pendatapan_to_angsuran'] >= 0,:]
df = df.reset_index(drop=True)
df['pendidikan'] = df['pendidikan'].map(lambda x: 'TidakSekolah' if x=='TIDAK SEKOLAH' else x)

"""
df['lama_bekerja'] = df.lama_bekerja.map(lambda x: 
        '<2'   if (x<2) else (
        '2-6'   if (x>=2) & (x<7) else (
        '7-10'   if (x>=7) & (x<11) else (
        '>=11'   if x>=11 else None
        )))
    )

df['jenis_usaha_perusahaan'] = df.jenis_usaha_perusahaan.map(lambda x: 
        x.rstrip().lstrip()   if x.rstrip().lstrip() in ['Riset & Pengembangan','Penganggur/Belum Bekerja/Pencari Kerja','Komputer-Analis/Progrmr/Internt/MIS/Engi'] else (
        'LainLain' if x.strip() > '' else (
        None
        ))
    )

df['propinsi'] = df.propinsi.map(lambda x: 
        '|'.join(['GORONTALO','SULAWESI TENGAH','SULAWESI UTARA'])   if x.rstrip().lstrip() in ['GORONTALO','SULAWESI TENGAH','SULAWESI UTARA'] else (
        '|'.join(['LAMPUNG','KEP BANGKA BLT'])   if x.rstrip().lstrip() in ['LAMPUNG','KEP BANGKA BLT'] else (
        '|'.join(['NANGGROE ACEH DARUSSALAM','PAPUA','NUSA TENGGARA BARAT']) if x.rstrip().lstrip() in ['NANGGROE ACEH DARUSSALAM','PAPUA','NUSA TENGGARA BARAT'] else (
        '|'.join(['JAWA BARAT','SUMATERA SELATAN','BANTEN','MALUKU']) if x.rstrip().lstrip() in ['JAWA BARAT','SUMATERA SELATAN','BANTEN','MALUKU'] else (
        '|'.join(['RIAU','JAMBI','JAWA TENGAH','BENGKULU']) if x.rstrip().lstrip() in ['RIAU','JAMBI','JAWA TENGAH','BENGKULU'] else (
        '|'.join(['BALI']) if x.rstrip().lstrip() in ['BALI'] else (
        'LainLain' if x.strip() > '' else (
        None
        )))))))
    )

df.apply(lambda x: x.count(), axis=0)

df['usia'] = df.usia.map(lambda x: 
        '<20'   if x<20 else (
        '20-25' if (x>=20) & (x<=25) else (
        '26-31' if (x>=26) & (x<=31) else (
        '32-41' if (x>=32) & (x<=41) else (
        '42-66' if (x>=42) & (x<=66) else (
        '67-70' if (x>=67) & (x<=70) else (
        '>70'   if x>70 else None
        ))))))
    )

df['usia'] = df.usia.map(lambda x: 
        1 if (x == '<20') else (
        2 if (x == '20-25') else (
        3 if (x == '26-31') else (
        4 if (x == '32-41') else (
        5 if (x == '42-66') else (
        6 if (x == '67-70') else (
        7 if (x == '>70') else (
        )))))))
    )

df['jumlah_anak_dlm_tanggungan'] = df.jumlah_anak_dlm_tanggungan.map(lambda x: 
        '0'   if x==0 else (
        '1-2' if (x>=1) & (x<=2) else (
        '3-5' if (x>=3) & (x<=5) else (
        '6-8' if (x>=6) & (x<=8) else (
        '>8'   if x>8 else None
        ))))
    )

df['jumlah_anak_dlm_tanggungan'] = df.jumlah_anak_dlm_tanggungan.map(lambda x: 
        1 if (x == '0') else (
        2 if (x == '1-2') else (
        3 if (x == '3-5') else (
        4 if (x == '6-8') else (
        5 if (x == '>8') else (
        )))))
    )

df['pendidikan'] = df.pendidikan.map(lambda x: 
        'S3'   if x.strip()=='S3' else (
        'S2'   if x.strip()=='S2' else (
        'S1'   if x.strip()=='S1' else (
        'SD-SMA'   if x.strip() in ['SD','SMP','SMA'] else None
        )))
    )

df['pendidikan'] = df.pendidikan.map(lambda x: 
        1 if (x == 'S3') else (
        2 if (x == 'S2') else (
        3 if (x == 'S1') else (
        4 if (x == 'SD-SMA') else None)))
    )

df['status_bukti_kepemilikan_agunan'] = df.status_bukti_kepemilikan_agunan.map(lambda x: 
        'Lainlain'   if x=='Kwitansi' else x
    )

df['status_bukti_kepemilikan_agunan'] = df.status_bukti_kepemilikan_agunan.map(lambda x: 
        1 if (x == 'SHM') else (
        2 if (x == 'SHGB') else (
        3 if (x == 'BPKB') else (
        4 if (x == 'SuratTanahDiLuarSertifikat') else (
        5 if (x == 'Lainlain') else (
        )))))
    )

df['lama_bekerja'] = df.lama_bekerja.map(lambda x: 
        '0-1'   if (x>=0) & (x<=1) else (
        '2-4'   if (x>=2) & (x<=4) else (
        '5-6'   if (x>=5) & (x<=6) else (
        '7-9'   if (x>=7) & (x<=9) else (
        '10-17' if (x>=10) & (x<=17) else (
        '18-23' if (x>=18) & (x<=23) else (
        '24-32' if (x>=24) & (x<=32) else (
        '33-38' if (x>=33) & (x<=38) else (
        '39-44' if (x>=39) & (x<=44) else (
        '45-50' if (x>=45) & (x<=50) else (
        '>50'   if x>8 else None
        ))))))))))
    )

df['lama_bekerja'] = df.lama_bekerja.map(lambda x: 
        1 if (x == '0-1') else (
        2 if (x == '2-4') else (
        3 if (x == '5-6') else (
        4 if (x == '7-9') else (
        5 if (x == '10-17') else (
        6 if (x == '18-23') else (
        7 if (x == '24-32') else (
        8 if (x == '33-38') else (
        9 if (x == '39-44') else (
        10 if(x == '45-50') else (
        11 if(x == '>50') else (
        )))))))))))
    )

df['status_nikah'] = df.status_nikah.map(lambda x: 
        1 if (x == 'JandaDuda') else (
        2 if (x == 'BelumMenikah') else (
        3 if (x == 'Menikah') else (
        )))
    )

df['kepemilikan_tempat_tinggal'] = df.kepemilikan_tempat_tinggal.map(lambda x: 
        1 if (x == 'status0') else (
        2 if (x == 'status1') else (
        3 if (x == 'status2') else (
        4 if (x == 'status3') else (
        ))))
    )

df['tujuan_penggunaan'] = df.tujuan_penggunaan.map(lambda x: 
        1 if (x == 'KI') else (
        2 if (x == 'PMK') else (
        3 if (x == 'KMK') else (
        )))
    )

df['jenis_kelamin'] = df.jenis_kelamin.map(lambda x: 
        0 if (x == 'M') else (
        1 if (x == 'F') else (
        ))
    )

df['jangka_waktu'] = df.jangka_waktu.map(lambda x: 
        '<6'    if (x<6) else (
        '6-9'   if (x>=6) & (x<=9) else (
        '12'    if (x>=11) & (x<=12) else (
        '18' if (x>12) & (x<=18) else (
        '24' if (x>18) & (x<=24) else (
        '36' if (x>24) & (x<=36) else (
        '48' if (x>36) & (x<=48) else (
        '60' if (x>48) & (x<=60) else None
        )))))))
    )

df['jangka_waktu'] = df.jangka_waktu.map(lambda x: 
        0 if (x == '<6') else (
        1 if (x == '6-9') else (
        2 if (x == '12') else (
        3 if (x == '18') else (
        4 if (x == '24') else (
        5 if (x == '36') else (
        6 if (x == '48') else (
        7 if (x == '60') else None)))))))
    )

df['ratio_pendatapan_utama_to_angsuran'] = df.ratio_pendatapan_utama_to_angsuran.map(lambda x: 
        10 if (x > 10) else x
    )

df = df.dropna(subset=['ratio_pendatapan_utama_to_angsuran','jangka_waktu'])
df = df.drop(['suku_bunga','jenis_usaha_perusahaan','plafond'], axis=1)
df.to_csv(index=None, path_or_buf="./data/las_dataset_kupedes_clean_v3_eda_virt_v2.csv")
"""

df.to_csv(index=None, path_or_buf="./data/las_dataset_kupedes_clean_v3_eda_v2.csv")

df = pd.read_csv("./data/las_dataset_kupedes_clean_v3_eda_v2.csv")