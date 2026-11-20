import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# LOAD MODEL
model = load_model("model_makanan.keras")

# DAFTAR KELAS
kelas = {
    0: "Bakwan Sayur",
    1: "Bubur Tinotuan",
    2: "Nasi Beras Merah",
    3: "Plencing Kangkung",
    4: "Wortel"
}

# 6) LOAD GAMBAR tes0.jpeg
img = image.load_img(
    "tes0.jpeg",
    target_size=(224, 224)
)

# Ubah gambar menjadi array
img = image.img_to_array(img)

# Normalisasi
img = img / 255.0

# Tambahkan dimensi batch
img = np.expand_dims(img, axis=0)

# PREDIKSI
hasil = model.predict(img)

# Ambil index dengan nilai terbesar
index = np.argmax(hasil[0])

# Ambil nilai probabilitas
nilai = hasil[0][index]

# HASIL
print("Nilai :", nilai)
print("Index kelas :", index)
print("Kelas :", kelas[index])

# 7) Pengujian kembali untuk citra yang lain (tes1.jpeg s/d tes4.jpeg)
for i in range(1, 5):
    fname = f"tes{i}.jpeg"
    if os.path.exists(fname):
        print(f"\nPengujian untuk {fname}:")
        img_n = image.load_img(fname, target_size=(224, 224))
        img_n = image.img_to_array(img_n) / 255.0
        img_n = np.expand_dims(img_n, axis=0)
        
        hasil_n = model.predict(img_n)
        index_n = np.argmax(hasil_n[0])
        nilai_n = hasil_n[0][index_n]
        
        print("Nilai :", nilai_n)
        print("Index kelas :", index_n)
        print("Kelas :", kelas[index_n])
