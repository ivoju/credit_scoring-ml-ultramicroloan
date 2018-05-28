# Machine Learning Ultra Micro Loan Credit Scoring Micro Service

This program using python3

Run Program
```
python3 main.py
```

Request Spesification
```
url: http://localhost:5000//ml_ultramicroloan
heder: Content-Type:application/json
method: POST
body: {
  "jangka_waktu":24,
  "plafond": 25000000.0,
  "jenis_kelamin": "M",
  "pendidikan":"SD/SMP/SMU/SMK",
  "status_nikah": "BelumMenikah",
  "jumlah_anak_dlm_tanggungan": 1,
  "kepemilikan_tempat_tinggal": "MilikSendiri",
  "debitur_lama": 1.0,
  "tujuan_penggunaan": "KI",
  "lama_bekerja": 1,
  "simpanan": 1238400.0,
  "usia": 71.0,
  "ratio_pendatapan_to_angsuran": 1.41481,
  "punya_usaha_sampingan": 1.0
}
```

Response Spesification
```
{
    "pd": "0.10587982833385468",
    "rating": "4",
    "score": "788.015698326779",
    "status": "OK"
}
```
