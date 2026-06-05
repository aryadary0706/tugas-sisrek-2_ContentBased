# 🎬 Tugas Kelompok Ke-2: Content BasedRecommendation System

Proyek ini merupakan sistem rekomendasi berbasis **Content-Based Filtering (User-Based Profile)** yang dikembangkan untuk memenuhi **Tugas Kelompok 2 - Sistem Rekomendasi**. Sistem ini memanfaatkan representasi semantik tingkat tinggi menggunakan **Sentence-BERT (SBERT)** untuk menghasilkan rekomendasi film atau *TV show* yang relevan berdasarkan riwayat tontonan (*watch history*) dan preferensi kategori pengguna.

---

## 👥 Anggota Kelompok 2
* **M. Paksi Pratama** - 103012300432
* **Kemas M. Aryadary Rasyad** - 103012300176
* **Atha Imtinan Ukirhayekti Larasati** - 103012300183
* **Raisul Gufran** - 103012300417

---

## 📌 Struktur Repositori & Alur Kerja
* **`tugaskelompok2_recsys.ipynb`**: Digunakan sebagai lingkungan eksperimen awal, analisis data eksploratif, serta pengujian logika komputasi kasar (sandbox) sebelum diimplementasikan ke sistem utama.
* **`app.py`**: Berkas utama aplikasi web berbasis **Streamlit**. Berkas ini mengimplementasikan GUI interaktif, manajemen profil pengguna (termasuk 12 skenario pengujian kasus ekstrem seperti *cold-start*), dan visualisasi metrik evaluasi secara langsung.

---

## 📊 Dataset Konten
Sistem ini menggunakan dataset publik Netflix metadata yang mencakup informasi detail mengenai film dan serial televisi.
* **Sumber Dataset:** [Kaggle - Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)
* **Berkas yang Diperlukan:** `netflix_titles.csv` (letakkan di root direktori yang sama dengan `app.py`).

---

## 🛠️ Pemasangan & Persiapan Lingkungan

### 1. Prasyarat Pustaka (Libraries)
Proyek ini menggunakan beberapa pustaka utama untuk pemrosesan data, machine learning, dan antarmuka pengguna:
* `streamlit` (Framework GUI Web)
* `pandas` & `numpy` (Manipulasi Data dan Komputasi Matriks)
* `scikit-learn` (Komputasi *Cosine Similarity*)
* `sentence-transformers` (Model Arsitektur SBERT Deep Learning)

### 2. Cara Instalasi
Buka terminal/command prompt pada direktori proyek, lalu jalankan perintah berikut untuk memasang seluruh dependensi:

```bash
pip install streamlit pandas numpy scikit-learn sentence-transformers