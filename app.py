import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Digital Twin System", layout="wide")
st.title("🧠 Digital Twin + Predictive Intelligence System")

# -----------------------------
# MODEL
# -----------------------------


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# -----------------------------
# DATASET
# -----------------------------


def create_dataset(data, time_step=5):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step])
    return np.array(X), np.array(y)


# -----------------------------
# INIT SESSION
# -----------------------------
if "data" not in st.session_state:
    st.session_state.data = []
    st.session_state.model = LSTMModel()
    st.session_state.scaler = MinMaxScaler()
    st.session_state.optimizer = torch.optim.Adam(
        st.session_state.model.parameters(), lr=0.001)
    st.session_state.criterion = nn.MSELoss()
    st.session_state.anomaly_count = 0

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("🎮 Control System")

temp = st.slider("Temperature", -10.0, 10.0, 0.0)
pressure = st.slider("Pressure", -10.0, 10.0, 0.0)
vibration = st.slider("Vibration", -10.0, 10.0, 0.0)

col1, col2 = st.columns(2)

with col1:
    if st.button("➕ Add Data"):
        st.session_state.data.append([temp, pressure, vibration])

with col2:
    if st.button("🔄 Reset"):
        st.session_state.data = []
        st.session_state.anomaly_count = 0

# -----------------------------
# MAIN LOGIC
# -----------------------------
data = np.array(st.session_state.data)

if len(data) > 7:

    scaled = st.session_state.scaler.fit_transform(data)

    X, y = create_dataset(scaled)

    # SAFETY CHECK
    if len(X) == 0:
        st.warning("Add more data")
        st.stop()

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    if len(X.shape) != 3:
        st.error("Data shape issue")
        st.stop()

    # TRAIN
    st.session_state.model.train()
    st.session_state.optimizer.zero_grad()
    output = st.session_state.model(X)
    loss = st.session_state.criterion(output, y)
    loss.backward()
    st.session_state.optimizer.step()

    # PREDICT
    last_input = torch.tensor(
        scaled[-5:].reshape(1, 5, 3), dtype=torch.float32)
    pred = st.session_state.model(last_input).detach().numpy()
    pred_real = st.session_state.scaler.inverse_transform(pred)[0]

    # -----------------------------
    # ANOMALY
    # -----------------------------
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    current = data[-1]

    anomaly = np.any(np.abs(current - mean) > 1.5 * std)

    if anomaly:
        st.session_state.anomaly_count += 1

    # -----------------------------
    # HEALTH
    # -----------------------------
    health = 100 - (40 if anomaly else 0)

    # -----------------------------
    # CONFIDENCE SCORE (NEW)
    # -----------------------------
    confidence = max(0, 100 - loss.item() * 100)

    # -----------------------------
    # TREND DETECTION (NEW)
    # -----------------------------
    trend = "Stable"
    if data[-1][0] > data[-2][0]:
        trend = "Increasing 📈"
    elif data[-1][0] < data[-2][0]:
        trend = "Decreasing 📉"

    # -----------------------------
    # GRAPH
    # -----------------------------
    fig, ax = plt.subplots()

    ax.plot(data[:, 0], label="Temperature")
    ax.plot(data[:, 1], label="Pressure")
    ax.plot(data[:, 2], label="Vibration")

    if anomaly:
        ax.scatter(len(data)-1, data[-1][0], s=120, label="Anomaly")

    ax.legend()
    ax.set_title("System Behavior")

    st.pyplot(fig)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("🔮 Prediction")

    st.write({
        "Temperature": round(pred_real[0], 2),
        "Pressure": round(pred_real[1], 2),
        "Vibration": round(pred_real[2], 2)
    })

    col1, col2, col3 = st.columns(3)

    col1.metric("❤️ Health", health)
    col2.metric("⚠️ Anomalies", st.session_state.anomaly_count)
    col3.metric("🎯 Confidence", round(confidence, 2))

    st.write("📊 Trend:", trend)

    if anomaly:
        st.error("⚠️ Anomaly Detected")
    else:
        st.success("✅ System Normal")

    # -----------------------------
    # DATA TABLE (NEW)
    # -----------------------------
    df = pd.DataFrame(data, columns=["Temp", "Pressure", "Vibration"])
    st.subheader("📋 Data History")
    st.dataframe(df)

    # -----------------------------
    # CSV DOWNLOAD (NEW)
    # -----------------------------
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Data", csv, "data.csv", "text/csv")

else:
    st.info("👉 Add at least 8 data points to activate AI")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "🚀 Advanced Digital Twin System with AI Prediction, Monitoring & Simulation")
