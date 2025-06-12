# app.py - Quantum Metals Predictor
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Configuración
st.set_page_config(page_title="Quantum Metals Predictor", page_icon="💰", layout="wide")
st.title("💰 Quantum Metals Predictor")
st.markdown("**Predicción de precios en tiempo real para Oro (XAU) y Plata (XAG)**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    timeframe = st.selectbox("Período de predicción", ["1 día", "3 días", "1 semana"])
    show_gold = st.checkbox("Mostrar Oro (XAU)", True)
    show_silver = st.checkbox("Mostrar Plata (XAG)", True)
    st.divider()
    st.markdown("📈 Datos obtenidos en tiempo real desde Yahoo Finance")

# Descargar datos reales
@st.cache_data
def get_data():
    end = datetime.today()
    start = end - timedelta(days=365)
    xau = yf.download("GC=F", start=start, end=end)  # Gold Futures
    xag = yf.download("SI=F", start=start, end=end)  # Silver Futures
    df = pd.DataFrame({
        "Date": xau.index,
        "XAU": xau["Adj Close"],
        "XAG": xag["Adj Close"]
    }).dropna()
    return df

df = get_data()

# Visualización
st.header("📊 Precios Históricos")
if show_gold:
    st.plotly_chart(px.line(df, x="Date", y="XAU", title="Precio Oro (XAU/USD)"), use_container_width=True)
if show_silver:
    st.plotly_chart(px.line(df, x="Date", y="XAG", title="Precio Plata (XAG/USD)"), use_container_width=True)

# Preparación de datos para LSTM
def prepare_lstm_data(series, window=30):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled) - window):
        X.append(scaled[i:i+window])
        y.append(scaled[i+window])
    return np.array(X), np.array(y), scaler

# Entrenamiento y predicción
st.header("🤖 Predicción con LSTM (Deep Learning)")

activo = st.selectbox("Activo a predecir", ["Oro (XAU)", "Plata (XAG)"])
target = "XAU" if activo.startswith("Oro") else "XAG"
series = df[target]

with st.spinner("Entrenando modelo LSTM..."):
    X, y, scaler = prepare_lstm_data(series)
    model = Sequential()
    model.add(LSTM(64, activation='tanh', input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    # Predicción futura
    last_sequence = series.values[-30:]
    last_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    pred_input = last_scaled.reshape((1, 30, 1))
    pred_scaled = model.predict(pred_input)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]

st.success(f"✅ Predicción del próximo valor de {target}: ${pred_price:.2f}")
st.metric(label=f"Valor actual de {target}", value=f"${series.values[-1]:.2f}")
st.metric(label="Predicción siguiente", value=f"${pred_price:.2f}")

# Footer
st.divider()
st.markdown("📌 **Modelo LSTM entrenado con datos reales.** App de demostración. Para producción, se recomienda recalibrar con múltiples variables y backtesting.")
