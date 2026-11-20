import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# LOAD MODEL
model = load_model("model_motor_mobil.keras")

# LOAD GAMBAR
img = image.load_img("tes1.jpeg", target_size=(224, 224))

# Ubah gambar menjadi array
img = image.img_to_array(img)

# Normalisasi
img = img / 255.0

# Tambahkan dimensi batch
img = np.expand_dims(img, axis=0)

# PREDIKSI
hasil = model.predict(img)

nilai = hasil[0][0]

print("Nilai prediksi :", nilai)

if nilai >= 0.5:
    print("Hasil : MOTOR")
else:
    print("Hasil : MOBIL")

# 5) Pengujian kembali untuk tes2.jpeg
print()
img2 = image.load_img("tes2.jpeg", target_size=(224, 224))
img2 = image.img_to_array(img2)
img2 = img2 / 255.0
img2 = np.expand_dims(img2, axis=0)
hasil2 = model.predict(img2)
nilai2 = hasil2[0][0]
print("Nilai prediksi :", nilai2)
if nilai2 >= 0.5:
    print("Hasil : MOTOR")
else:
    print("Hasil : MOBIL")
