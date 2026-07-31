import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Set Halaman Streamlit
st.set_page_config(
    page_title="FraudShield - Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# 1. Load Model & Artefak dari Colab
@st.cache_resource
def load_model_artifacts():
    saved_data = joblib.load("xgboost_fraud_model.pkl")
    return saved_data['model'], saved_data['features'], saved_data['label_encoders']

try:
    model, feature_names, label_encoders = load_model_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Gagal memuat model .pkl! Pastikan file 'xgboost_fraud_model.pkl' ada di folder yang sama.\nError: {e}")

# Navigation / Sidebar Menu
st.sidebar.title("🛡️ FraudShield")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Pilih Menu Navigation:", [
    "1. Real-Time Fraud Simulator", 
    "2. Batch Prediction (CSV)", 
    "3. Model Analytics & Performance"
])

# MENU 1: REAL-TIME FRAUD SIMULATOR
if menu == "1. Real-Time Fraud Simulator":
    st.title("⚡ Real-Time Transaction Fraud Simulator")
    st.write("Simulasikan transaksi baru secara interaktif untuk mengecek tingkat risiko fraud secara *real-time*.")

    if model_loaded:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("💳 Detail Transaksi")
            amount = st.number_input("Nominal Transaksi ($)", min_value=1.0, value=15000.0, step=100.0)
            credit_limit = st.number_input("Limit Kredit Kartu ($)", min_value=1000.0, value=300000.0, step=1000.0)
            annual_income = st.number_input("Pendapatan Tahunan ($)", min_value=1000.0, value=1000000.0, step=5000.0)
            age = st.slider("Umur Nasabah", 18, 90, 40)

        with col2:
            st.subheader("🌐 Lokasi & Channel")
            is_intl = st.selectbox("Transaksi Internasional?", ["Tidak", "Ya"])
            state_mismatch = st.selectbox("Provinsi Nasabah & Merchant Beda?", ["Tidak", "Ya"])
            city_mismatch = st.selectbox("Kota Nasabah & Merchant Beda?", ["Tidak", "Ya"])
            txn_hour = st.slider("Jam Transaksi (0 - 23)", 0, 23, 14)

        with col3:
            st.subheader("⚙️ Kategori & Metode")
            merchant_cat = st.selectbox("Kategori Merchant", [
                'Hotel', 'Travel', 'Fuel', 'Fashion', 'Airline', 'Hospital', 'Ecommerce', 'Entertainment', 'Food Delivery', 'Financial Services'
            ])
            card_type = st.selectbox("Tipe Kartu", ['Credit', 'Debit'])
            payment_method = st.selectbox("Metode Pembayaran", ['Chip', 'Online', 'Swipe', 'Contactless'])

        st.markdown("---")
        
        # Tombol Prediksi
        if st.button("🚨 Cek Risiko Transaksi", use_container_width=True):
            # Feature Engineering untuk Input
            amt_to_limit = amount / (credit_limit + 1)
            amt_to_income = amount / (annual_income + 1)
            is_weekend = 0 # Default simulasi
            is_night = 1 if (txn_hour >= 22 or txn_hour <= 5) else 0

            # Susun dictionary input sesuai kolom fitur saat training
            input_dict = {
                'Transaction_Amount': amount,
                'Credit_Limit': credit_limit,
                'Annual_Income': annual_income,
                'Age': age,
                'Is_International': 1 if is_intl == "Ya" else 0,
                'Is_State_Mismatch': 1 if state_mismatch == "Ya" else 0,
                'Is_City_Mismatch': 1 if city_mismatch == "Ya" else 0,
                'Amount_to_Limit_Ratio': amt_to_limit,
                'Amount_to_Income_Ratio': amt_to_income,
                'Transaction_Hour': txn_hour,
                'Is_Weekend': is_weekend,
                'Is_Night_Transaction': is_night,
                'Merchant_Category': merchant_cat,
                'Card_Type': card_type,
                'Payment_Method': payment_method
            }

            input_df = pd.DataFrame([input_dict])

            # Encode fitur kategorikal sesuai LabelEncoder Colab jika ada
            for col, le in label_encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = le.transform(input_df[col].astype(str))
                    except:
                        input_df[col] = 0

            # Pastikan urutan kolom persis sama dengan urutan saat training
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_names]

            # Prediksi
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1] * 100

            # Tampilan Hasil
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.metric("Skor Risiko Fraud", f"{probability:.2f}%")
            
            with res_col2:
                if prediction == 1 or probability > 50:
                    st.error("🔴 **ALERT: TRANSAKSI HIGH RISK (INDIKASI FRAUD DETECTED)**")
                    st.write("Sistem menyarankan untuk **MEMBLOKIR** sementara transaksi ini dan mengirimkan OTP verifikasi ke nasabah.")
                else:
                    st.success("🟢 **SUCCESS: TRANSAKSI LOW RISK (TRANSAKSI AMAN)**")
                    st.write("Transaksi memenuhi kriteria perilaku normal nasabah dan dapat langsung diproses.")

# MENU 2: BATCH PREDICTION (CSV)
elif menu == "2. Batch Prediction (CSV)":
    st.title("📂 Batch Fraud Analytics (File CSV)")
    st.write("Upload file CSV transaksi untuk melakukan pemindaian massal otomatis.")

    uploaded_file = st.file_uploader("Unggah File CSV Transaksi", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.subheader("📌 Preview Data yang Diunggah")
        st.dataframe(df_uploaded.head())

        if st.button("🚀 Jalankan Pemindaian Batch"):
            st.info("Fitur pemindaian batch siap memproses file berdasarkan model XGBoost.")

# MENU 3: MODEL ANALYTICS & PERFORMANCE
elif menu == "3. Model Analytics & Performance":
    st.title("📊 Model Performance & Risk Analytics")
    st.write("Evaluasi statistik performa model XGBoost dalam mendeteksi penipuan.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Akurasi Model", "94.86%")
    m2.metric("Recall (Fraud)", "92.36%", delta="Target Utama", delta_color="normal")
    m3.metric("Precision (Fraud)", "51.29%")
    m4.metric("F1-Score", "65.95%")

    st.markdown("---")
    st.subheader("💡 Ringkasan Evaluasi Performa")
    st.write("""
    - **Recall Tinggi (92.36%):** Model berhasil menangkap 2.489 dari total 2.695 transaksi fraud sungguhan.
    - **False Negative Rendah (206 kasus):** Kebobolan transaksi fraud berhasil ditekan hingga batas minimal.
    - **False Positive (2.364 kasus):** Trade-off sensitivitas penyeimbangan kelas (*scale_pos_weight*) untuk menjaga uang nasabah tetap aman.
    """)