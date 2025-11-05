import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === CONFIGURASI STREAMLIT ===
st.set_page_config(page_title="Prediksi Kepuasan Pelanggan", page_icon="📊", layout="wide")

st.title("📊 Prediksi Kepuasan Pelanggan")
st.markdown("Aplikasi ini memprediksi tingkat kepuasan pelanggan (skala 1–5) berdasarkan data pelanggan menggunakan model *Random Forest Regressor*.")

# === LOAD DATASET ===
uploaded_file = st.file_uploader("📂 Upload dataset CSV (customer_satisfaction.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset berhasil dimuat!")
    
    # Tampilkan contoh data
    st.subheader("📋 Contoh Data")
    st.dataframe(df.head())

    # === Preprocessing ===
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])  # Male=1, Female=0

    X = df.drop('Satisfaction', axis=1)
    y = df['Satisfaction']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === Train Model ===
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X_scaled, y)

    # === Evaluasi Model ===
    y_pred = model.predict(X_scaled)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    st.subheader("📈 Evaluasi Model")
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")

    # === Feature Importance ===
    feat_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    st.subheader("🔍 Feature Importance")
    st.bar_chart(feat_importance.set_index('Feature'))

    # === Input Prediksi Baru ===
    st.subheader("🧍 Prediksi Kepuasan Pelanggan Baru")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Umur", min_value=18, max_value=60, value=30)
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    with col2:
        spend = st.number_input("Pengeluaran Bulanan (Rp)", min_value=100000, max_value=1000000, value=450000)
        usage = st.slider("Frekuensi Penggunaan Layanan (1–5)", 1, 5, 4)
    with col3:
        tickets = st.number_input("Jumlah Tiket Dukungan", min_value=0, max_value=10, value=1)
        response = st.number_input("Waktu Respon (jam)", min_value=0.5, max_value=10.0, value=2.0)

    if st.button("🔮 Prediksi Kepuasan"):
        new_data = pd.DataFrame({
            'Age': [age],
            'Gender': [1 if gender == "Male" else 0],
            'Monthly_Spend': [spend],
            'Service_Usage': [usage],
            'Support_Tickets': [tickets],
            'Response_Time': [response]
        })
        new_scaled = scaler.transform(new_data)
        prediction = model.predict(new_scaled)[0]
        st.success(f"⭐ Prediksi Skor Kepuasan Pelanggan: **{prediction:.2f} / 5.00**")
else:
    st.info("⬆️ Silakan upload file `customer_satisfaction.csv` terlebih dahulu.")
