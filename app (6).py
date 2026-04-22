import streamlit as st
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib

st.set_page_config(page_title="Battery Dashboard", layout="wide")

st.title("🔋 Battery SOC Prediction Dashboard")

# =========================
# LOAD MODEL
# =========================
model = load_model("ann_model_final.h5", compile=False)

# Load scaler
try:
    scaler = joblib.load("scaler.pkl")
except:
    scaler = None
    st.warning("Scaler not found!")

# =========================
# SESSION STATE (for graphs)
# =========================
if "voltage_hist" not in st.session_state:
    st.session_state.voltage_hist = []
    st.session_state.current_hist = []
    st.session_state.soc_hist = []

# =========================
# USER INPUT
# =========================
st.sidebar.header("Enter Battery Values")

voltage = st.sidebar.number_input("Voltage (V)", 0.0, 20.0, 12.0)
current = st.sidebar.number_input("Current (A)", -50.0, 50.0, 0.0)

# =========================
# PREDICTION
# =========================
input_data = np.array([[current, voltage]])

if scaler:
    input_data = scaler.transform(input_data)

soc = model.predict(input_data)[0][0]
soc = max(0, min(100, soc))

# =========================
# STORE HISTORY
# =========================
st.session_state.voltage_hist.append(voltage)
st.session_state.current_hist.append(current)
st.session_state.soc_hist.append(soc)

# Limit history length
max_len = 50
st.session_state.voltage_hist = st.session_state.voltage_hist[-max_len:]
st.session_state.current_hist = st.session_state.current_hist[-max_len:]
st.session_state.soc_hist = st.session_state.soc_hist[-max_len:]

# =========================
# STATUS
# =========================
if current < 0:
    status = "⚡ Charging"
    color = "green"
elif current > 0:
    status = "🔻 Discharging"
    color = "red"
else:
    status = "⏸ Idle"
    color = "gray"

st.markdown(f"## Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)

# =========================
# GAUGE FUNCTION
# =========================
def create_gauge(title, value, min_val, max_val, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': color},
        }
    ))
    return fig

# =========================
# GAUGES
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.plotly_chart(create_gauge("Voltage (V)", voltage, 0, 15, "blue"))

with col2:
    st.plotly_chart(create_gauge("Current (A)", current, -50, 50, "orange"))

with col3:
    st.plotly_chart(create_gauge("SOC (%)", soc, 0, 100, "green"))

# =========================
# GRAPHS
# =========================
st.write("## 📊 Real-Time Graphs")

col4, col5, col6 = st.columns(3)

with col4:
    st.line_chart(st.session_state.voltage_hist)

with col5:
    st.line_chart(st.session_state.current_hist)

with col6:
    st.line_chart(st.session_state.soc_hist)