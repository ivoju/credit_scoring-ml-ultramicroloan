import pandas as pd

pd.options.display.html.table_schema = True
pd.options.display.max_columns = 99

from model import *

from flask import Flask, jsonify
from flask import abort
from flask import request
from flask import json

app = Flask(__name__)

#@app.route('/')
@app.route('/ml_ultramicroloan', methods=['POST'])
def create_task():
    #if not request.json or not 'title' in request.json:
    if not request.json:
        abort(400)

    ## -- Parsing data
    data = {
        'tgl_mulai': ['1999-01-01'],
        'jangka_waktu': int(request.json['jangka_waktu']),
        'plafond': float(request.json['plafond']),
        'jenis_kelamin': str(request.json['jenis_kelamin']),
        'pendidikan': str(request.json['pendidikan']),
        'status_nikah': str(request.json['status_nikah']),
        'jumlah_anak_dlm_tanggungan': int(request.json['jumlah_anak_dlm_tanggungan']),
        'kepemilikan_tempat_tinggal': str(request.json['kepemilikan_tempat_tinggal']),
        'debitur_lama': float(request.json['debitur_lama']),
        'tujuan_penggunaan': str(request.json['tujuan_penggunaan']),
        'status_bukti_kepemilikan_agunan': ['XXX'],
        'lama_bekerja': int(request.json['lama_bekerja']),
        'pemasaran': [0.0],
        'simpanan': float(request.json['simpanan']),
        'usia': float(request.json['usia']),
        'ratio_pendatapan_to_angsuran': float(request.json['ratio_pendatapan_to_angsuran']),
        'punya_usaha_sampingan': float(request.json['punya_usaha_sampingan']),
        'ratio_likuidasi_agunan_to_plafond': [0.0],
        'target': [0],
    }

    ## -- Create data frame that equal for input model
    df_create = pd.DataFrame(data=data)

    ordered_cols = [
    	'tgl_mulai',
    	'jangka_waktu',
    	'plafond',
    	'jenis_kelamin',
    	'pendidikan',
    	'status_nikah',
    	'jumlah_anak_dlm_tanggungan',
    	'kepemilikan_tempat_tinggal',
    	'debitur_lama',
    	'tujuan_penggunaan',
    	'status_bukti_kepemilikan_agunan',
    	'lama_bekerja',
    	'pemasaran',
    	'simpanan',
    	'usia',
    	'ratio_pendatapan_to_angsuran',
    	'punya_usaha_sampingan',
    	'ratio_likuidasi_agunan_to_plafond',
    	'target'
    ]

    ## -- Order columns (to be equal to modeling step)
    df_create = df_create[ordered_cols]

    ## -- Read rule of credit scoring grade and transform prediction result to grade level
    df_grade = pd.read_csv("./model/las_kupedes_ultramikro_v3_cv_xgb_grade.csv")

    ## -- Doing scoring of customer data
    result = getscore_ultramicroloan(df_create, df_grade)

    return jsonify({'status': 'OK', 'pd': str(result.pd), 'score': str(result.score), 'rating': str(result.rating)}), 200

if __name__ == '__main__':
    app.run(debug=True)



# ## -- Create data frame that equal for input model
# data = {'tgl_mulai': ['1999-01-01'], 'jangka_waktu': [24], 'plafond': [25000000.0], 'jenis_kelamin': ['M'], 
#         'pendidikan': ['SD/SMP/SMU/SMK'], 'status_nikah': ['BelumMenikah'], 'jumlah_anak_dlm_tanggungan': [1], 
#         'kepemilikan_tempat_tinggal': ['MilikSendiri'], 'debitur_lama': [1.0], 'tujuan_penggunaan': ['KI'], 
#         'status_bukti_kepemilikan_agunan': ['XXX'], 'lama_bekerja': [1], 'pemasaran': [0.0], 'simpanan': [1238400.0], 
#         'usia': [71.0], 'ratio_pendatapan_to_angsuran': [1.41481], 'punya_usaha_sampingan': [1.0], 'ratio_likuidasi_agunan_to_plafond': [0.0], 'target': [0]}

# df_create = pd.DataFrame(data=data)

# ## -- Order columns (to be equal to modeling step)
# df_create = df_create[['tgl_mulai', 'jangka_waktu', 'plafond', 'jenis_kelamin', 'pendidikan', 'status_nikah', 
#                         'jumlah_anak_dlm_tanggungan', 'kepemilikan_tempat_tinggal', 'debitur_lama', 'tujuan_penggunaan', 
#                         'status_bukti_kepemilikan_agunan', 'lama_bekerja', 'pemasaran', 'simpanan', 'usia', 'ratio_pendatapan_to_angsuran', 
#                         'punya_usaha_sampingan', 'ratio_likuidasi_agunan_to_plafond', 'target']]

# ## -- Read rule of credit scoring grade and transform prediction result to grade level
# df_grade = pd.read_csv("./model/las_kupedes_ultramikro_v3_cv_xgb_grade.csv")

# ## -- Doing scoring of customer data
# result = getscore_ultramicroloan(df_create, df_grade)


# print("RESULT: ", result.score)