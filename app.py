import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Manual Digital Twin", layout="wide")

st.title("🧠 Digital Twin (Manual Input System)")

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
    for i in range(len(data)-time_step-1):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step])
    return np.array(X), np.array(y)


# -----------------------------
# INIT
# -----------------------------
if "data" not in st.session_state:
    st.session_state.data = []
    st.session_state.model = LSTMModel()
    st.session_state.scaler = MinMaxScaler()
    st.session_state.optimizer = torch.optim.Adam(
        st.session_state.model.parameters(), lr=0.001)
    st.session_state.criterion = nn.MSELoss()

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("🎮 Control Your System")

temp = st.slider("Temperature", -10.0, 10.0, 0.0)
pressure = st.slider("Pressure", -10.0, 10.0, 0.0)
vibration = st.slider("Vibration", -10.0, 10.0, 0.0)

if st.button("Add Data Point"):
    st.session_state.data.append([temp, pressure, vibration])

# -----------------------------
# PROCESS DATA
# -----------------------------
data = np.array(st.session_state.data)

if len(data) > 5:
    scaled = st.session_state.scaler.fit_transform(data)

    X, y = create_dataset(scaled)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # TRAIN
    st.session_state.model.train()
    st.session_state.optimizer.zero_grad()
    output = st.session_state.model(X)
    loss = st.session_state.criterion(output, y)
    loss.backward()
    st.session_state.optimizer.step()

    # PREDICT NEXT
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

    anomaly = np.any(np.abs(current - mean) > 1.5*std)

    # -----------------------------
    # HEALTH
    # -----------------------------
    health = 100 - (40 if anomaly else 0)

    # -----------------------------
    # PLOT
    # -----------------------------
    fig, ax = plt.subplots()

    ax.plot(data[:, 0], label="Temp")
    ax.plot(data[:, 1], label="Pressure")
    ax.plot(data[:, 2], label="Vibration")

    ax.scatter(len(data)-1, data[-1][0], s=100)

    ax.legend()
    ax.set_title("Your Digital Twin")

    st.pyplot(fig)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("🔮 Prediction")
    st.write("Next State:", pred_real)

    st.metric("Health Score", health)

    if anomaly:
        st.error("⚠️ Anomaly Detected")
    else:
        st.success("System Normal")

else:
    st.info("Add at least 6 data points to start prediction")
