# Kecerdasan Buatan - Pertemuan 11: Deep Learning 1 (CNN)

Repositori ini berisi seluruh implementasi praktikum, latihan, tugas kelompok, dan laporan komprehensif untuk **Bab 11: Pembelajaran Mendalam (Deep Learning 1)** menggunakan **Convolutional Neural Network (CNN)** dengan TensorFlow & Keras.

---

## 🚀 Jalankan Langsung di Google Colab (1-Klik T4 GPU)

Klik tombol di bawah ini untuk langsung membuka notebook di Google Colab:

| Materi Praktikum | Buka di Google Colab |
| :--- | :--- |
| **Praktikum 11.1: Klasifikasi Citra CNN (Mobil vs Motor)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shafw00n/Modul-11-KB/blob/main/Praktikum%2011.1.ipynb) |
| **Praktikum 11.2: Pemisahan Modul Train & Test 2 Kelas** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shafw00n/Modul-11-KB/blob/main/Praktikum%2011.2.ipynb) |
| **Praktikum 11.3: Klasifikasi Citra Multi-Kelas (5 Kelas Makanan)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shafw00n/Modul-11-KB/blob/main/Praktikum%2011.3.ipynb) |
| **Latihan: Klasifikasi Daun Tanaman (*Plant Leaves*)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akfaulin/Kecerdasan-Buatan-Pertemuan-11/blob/main/Latihan.ipynb) |
| **Tugas: Sistem Login Verifikasi Wajah 3 Mahasiswa** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akfaulin/Kecerdasan-Buatan-Pertemuan-11/blob/main/Tugas.ipynb) |

> **Catatan Penggunaan Colab:**  
> 1. Aktifkan GPU gratis di menu: **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ pilih **T4 GPU** $\rightarrow$ **Save**.  
> 2. Seluruh dataset akan otomatis terunduh dan terpasang saat cell pertama dijalankan.

---

## 📁 Struktur Direktori Repositori

```
├── Dataset 1/                      # Dataset citra Mobil dan Motor (Praktikum 11.1)
├── Dataset 2/                      # Dataset citra & citra uji tes1, tes2 (Praktikum 11.2)
├── Dataset 3/                      # Dataset citra 5 kelas makanan & tes0-tes4 (Praktikum 11.3)
├── Praktikum 11.1/                 # Folder proyek Praktikum 11.1 (klasifikasi.py)
├── Praktikum 11.2/                 # Folder proyek Praktikum 11.2 (train.py, test.py)
├── Praktikum 11.3/                 # Folder proyek Praktikum 11.3 (train.py, test.py)
├── Latihan/                        # Folder pengerjaan Latihan Kaggle Plant Leaves
├── Tugas/                          # Folder Tugas Sistem Login Wajah 3 Mahasiswa
├── Praktikum 11.1.ipynb            # Jupyter Notebook Praktikum 11.1
├── Praktikum 11.2.ipynb            # Jupyter Notebook Praktikum 11.2
├── Praktikum 11.3.ipynb            # Jupyter Notebook Praktikum 11.3
├── Latihan.ipynb                   # Jupyter Notebook Latihan
├── Tugas.ipynb                     # Jupyter Notebook Tugas
├── Modul_11_Praktikum_Lengkap.ipynb# Notebook Gabungan Seluruh Modul dengan Output Lengkap
└── Laporan_Praktikum_Modul_11.md   # Laporan Resmi Praktikum (Standar Akademik)
```

---

## 📊 Hasil Pengujian Model

### Praktikum 11.2 (Citra Uji 2 Kelas)
- `tes1.jpeg` $\rightarrow$ **MOBIL** (Probabilitas: `0.000330`, Target: $< 0.5$)
- `tes2.jpeg` $\rightarrow$ **MOTOR** (Probabilitas: `0.891319`, Target: $\ge 0.5$)

### Praktikum 11.3 (Citra Uji 5 Kelas Makanan)
- `tes0.jpeg` $\rightarrow$ **Bakwan Sayur** (Probabilitas: `97.70%`, Indeks 0)
- `tes1.jpeg` $\rightarrow$ **Bubur Tinotuan** (Probabilitas: `69.52%`, Indeks 1)
- `tes2.jpeg` $\rightarrow$ **Nasi Beras Merah** (Probabilitas: `92.84%`, Indeks 2)
- `tes3.jpeg` $\rightarrow$ **Plencing Kangkung** (Probabilitas: `98.49%`, Indeks 3)
- `tes4.jpeg` $\rightarrow$ **Wortel** (Probabilitas: `100.00%`, Indeks 4)

*Tingkat akurasi pengujian: 100% pada seluruh citra uji.*
