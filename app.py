import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# PAGE SETUP
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
# DATASET CREATION
# -----------------------------


def create_dataset(data, time_step=5):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step])
    return np.array(X), np.array(y)


# -----------------------------
# SESSION INIT
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
st.subheader("🎮 Control System (Manual Input)")

temp = st.slider("Temperature", -10.0, 10.0, 0.0)
pressure = st.slider("Pressure", -10.0, 10.0, 0.0)
vibration = st.slider("Vibration", -10.0, 10.0, 0.0)

col1, col2 = st.columns(2)

with col1:
    if st.button("➕ Add Data Point"):
        st.session_state.data.append([temp, pressure, vibration])

with col2:
    if st.button("🔄 Reset Data"):
        st.session_state.data = []
        st.session_state.anomaly_count = 0

# -----------------------------
# PROCESSING
# -----------------------------
data = np.array(st.session_state.data)

if len(data) > 5:

    # Scale
    scaled = st.session_state.scaler.fit_transform(data)

    # Dataset
    X, y = create_dataset(scaled)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Train
    st.session_state.model.train()
    st.session_state.optimizer.zero_grad()
    output = st.session_state.model(X)
    loss = st.session_state.criterion(output, y)
    loss.backward()
    st.session_state.optimizer.step()

    # Predict next
    last_input = torch.tensor(
        scaled[-5:].reshape(1, 5, 3), dtype=torch.float32)
    pred = st.session_state.model(last_input).detach().numpy()
    pred_real = st.session_state.scaler.inverse_transform(pred)[0]

    # -----------------------------
    # ANOMALY DETECTION
    # -----------------------------
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    current = data[-1]

    anomaly = np.any(np.abs(current - mean) > 1.5 * std)

    if anomaly:
        st.session_state.anomaly_count += 1

    # -----------------------------
    # HEALTH SCORE
    # -----------------------------
    health = 100 - (40 if anomaly else 0)

    # -----------------------------
    # GRAPH
    # -----------------------------
    fig, ax = plt.subplots()

    ax.plot(data[:, 0], label="Temperature")
    ax.plot(data[:, 1], label="Pressure")
    ax.plot(data[:, 2], label="Vibration")

    # Highlight anomaly
    if anomaly:
        ax.scatter(len(data)-1, data[-1][0], s=120, label="Anomaly")

    ax.legend()
    ax.set_title("System Behavior")

    st.pyplot(fig)

    # -----------------------------
    # OUTPUT SECTION
    # -----------------------------
    st.subheader("🔮 Prediction")

    st.write("Next Predicted State:")
    st.write({
        "Temperature": round(pred_real[0], 2),
        "Pressure": round(pred_real[1], 2),
        "Vibration": round(pred_real[2], 2)
    })

    col1, col2 = st.columns(2)

    with col1:
        st.metric("❤️ Health Score", health)

    with col2:
        st.metric("⚠️ Total Anomalies", st.session_state.anomaly_count)

    if anomaly:
        st.error("⚠️ Anomaly Detected! System behaving abnormally")
    else:
        st.success("✅ System Normal")

else:
    st.info("👉 Add at least 6 data points to start AI prediction")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "💡 This system learns your inputs and predicts future system behavior using AI.")
