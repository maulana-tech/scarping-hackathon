# CoreTax Sentiment Analysis

Project ini bertujuan untuk menganalisis sentimen publik terhadap sistem CoreTax DJP menggunakan data dari YouTube, Play Store, dan Media Sosial.

## Struktur Folder

```
project/
├── data/           -> Dataset mentah (raw) dan hasil preprocessing (processed)
│   ├── processed/  -> Data yang sudah dibersihkan dan digabungkan
│   └── ...         -> Data mentah (CSV scraping)
├── notebooks/      -> Jupyter Notebooks untuk analisis dan eksperimen
│   └── Hackathon Sentiment Analysis Improved.ipynb
├── src/            -> Source code utama dan script utilitas
│   └── verify_integration.py
├── models/         -> (Optional) Tempat menyimpan model yang sudah dilatih
├── outputs/        -> Hasil analisis, laporan, dan outline presentasi
│   └── presentation_outline.md
├── requirements.txt -> Daftar library Python yang dibutuhkan
└── README.md       -> Dokumentasi project ini
```

## Cara Menjalankan

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Jalankan Notebook:**
    Buka `notebooks/Hackathon Sentiment Analysis Improved.ipynb` menggunakan Jupyter Notebook atau Google Colab.
    Pastikan path data sudah sesuai (notebook sudah dikonfigurasi untuk mencari data di folder `../data/`).

3.  **Verifikasi Data:**
    Anda dapat menjalankan script verifikasi untuk mengecek integritas data:
    ```bash
    cd src
    python verify_integration.py
    ```

## Fitur Utama
- **Data Integration:** Menggabungkan data dari berbagai sumber.
- **Sentiment Analysis:** Menggunakan RoBERTa (`w11wo/indonesian-roberta-base-sentiment-classifier`).
- **Advanced Visualization:** WordCloud, N-grams, Co-occurrence Network.
- **Automated Insights:** Rekomendasi perbaikan berdasarkan analisis sentimen negatif.
