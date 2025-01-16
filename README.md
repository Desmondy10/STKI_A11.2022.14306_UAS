# **Analisis Sentimen Twitter Menggunakan Neural Network dengan TensorFlow**

 Analisis sentiment Twitter menggunakan neural network dengan TensorFlow menawarkan pendekatan yang canggih dan efektif dalam mengolah data teks besar dan memahami opini public dengan lebih baik. Dengan terus mengembangkan dan mengoptimalkan model-model ini, kita dapat memberikan kontribusi yang signifikan dalam mendapatkan wawasan berharga dari data teks yang dihasilkan oleh pengguna media social. Analisis sentiment terhadap tweet di Twitter sangat penting untuk memahami pandangan dan perasaan pengguna terhadap berbagai topik, produk, atau peristiwa tertentu.

## Dataset
Dataset yang digunakan adalah sebuah file CSV dengan nama training.1600000.processed.noemoticon.csv, yang berisi data tweet yang sudah diproses. Dataset ini digunakan untuk analisis sentimen dengan label "positive" atau "negative". Setiap baris dalam dataset berisi kolom-kolom berikut:

    1. label: Label sentimen dari tweet tersebut (positif atau negatif).
    2. time: Waktu tweet tersebut diposting.
    3. date: Tanggal tweet diposting.
    4. query: Jenis query atau pencarian yang digunakan untuk memperoleh tweet.
    5. username: Nama pengguna yang memposting tweet.
    6. text: Isi dari tweet itu sendiri (teks yang akan dianalisis).

## Preprocessing Data
    1. Penghapusan tanda baca, simbol, angka, dan URL.
    2. Konversi teks ke huruf kecil.
    3. Pembersihan teks dari stopwords menggunakan NLTK.
    4. Tokenisasi teks dan padding sequence untuk memastikan input memiliki panjang yang sama.

## Pemodelan
**Model menggunakan LSTM dengan arsitektur:**
    - Embedding Layer: Untuk representasi kata.
    - LSTM Layer: Untuk menangkap dependensi temporal dalam teks.
    - Fully Connected Layer dengan ReLU Activation.
    - Softmax Output Layer untuk klasifikasi sentimen (positif/negatif).

**Model dilatih menggunakan:**
    - Loss Function: Categorical Crossentropy.
    - Optimizer: RMSProp.
    - Dataset dibagi menjadi 80% data training dan 20% data testing.

## Evaluasi Model
Performa Model dievaluasi dengan menggunakan
    1. Confusion Matrix
    2. ROC Curve
    3. Accuracy, Precision, Recall dan F1 Score 
    dengan hasil sebagai berikut

    Accuracy: 69.6%
    Precision: 69.3%
    Recall: 72.5%
    F1 Score: 70.9%

## Kesimpulan
**Pencapaian Project:**
    Model sentiment analysis berbasis LSTM berhasil dibangun dan dievaluasi.
    Model memiliki performa yang cukup baik untuk tugas awal dengan akurasi hampir 70%.

**Rekomendasi untuk Pengembangan Lebih Lanjut:**
    1. Optimasi Preprocessing: Tambahkan langkah seperti lemmatization atau handling emoji untuk meningkatkan kualitas data input.
    2. Hyperparameter Tuning: Eksperimen dengan parameter seperti learning rate, jumlah neuron, atau embedding dimension.
    3. Gunakan Model Pretrained: Implementasikan model pretrained seperti BERT untuk meningkatkan performa.
    4. Evaluasi dengan Dataset Baru: Uji model pada dataset baru untuk memastikan generalisasi.

**Potensi Implementasi:**
    1. Model ini dapat digunakan untuk analisis sentimen pada tweet atau teks lainnya untuk memahami opini publik dalam berbagai domain, seperti pemasaran, politik, atau layanan pelanggan.
