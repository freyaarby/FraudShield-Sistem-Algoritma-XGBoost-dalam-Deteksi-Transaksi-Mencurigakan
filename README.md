# 🛡️ FraudShield - Real-Time Banking Fraud Detection System

FraudShield adalah aplikasi berbasis Machine Learning yang dirancang untuk memantau, mendeteksi, dan menganalisis potensi penipuan (*fraud*) pada transaksi perbankan secara *real-time*.

---

## 📌 Features
- **⚡ Real-Time Fraud Simulator:** Memprediksi skor risiko transaksi tunggal secara instan.
- **📂 Batch Fraud Analytics:** Memindai file CSV riwayat transaksi harian untuk deteksi massal.
- **📊 Model Performance & Insights:** Menampilkan metrik performa model dan visualisasi sektor rawan penipuan.

---

## 🛠️ Tech Stack & Model Performance
- **Language & Frameworks:** Python, Streamlit, Pandas, Scikit-Learn
- **Machine Learning Model:** XGBoost Classifier (dinetralkan dari *class imbalance* memakai `scale_pos_weight`)
- **Key Metrics:**
  - **Recall (Fraud Detection Rate):** **92.36%** (Meminimalkan *False Negative* / Kebobolan)
  - **Overall Accuracy:** **94.86%**
