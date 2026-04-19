import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Demand Intelligence Engine", layout="wide")
st.title("🚀 Demand Forecasting System (Improved)")

# -----------------------------
# SINGLE PRODUCT INPUT
# -----------------------------
st.subheader("🛍️ Product Setup")

product = st.text_input("Enter Product Name", placeholder="e.g., Jacket, Ice Cream")

product_type = st.selectbox(
    "Select Product Seasonality",
    ["Winter Product", "Summer Product", "All-Season Product"]
)

# -----------------------------
# PRODUCT EFFECT (REALISTIC LOGIC)
# -----------------------------
product_effect_map = {
    "Winter Product": 1,
    "Summer Product": 1,
    "All-Season Product": 1
}

# -----------------------------
# DATA GENERATION (FIXED - NO RANDOM NOISE MODEL)
# -----------------------------
@st.cache_data
def generate_data(product_type):
    np.random.seed(42)
    days = 300

    date = pd.date_range(start="2023-01-01", periods=days)

    price = np.random.uniform(50, 150, days)
    season = np.random.choice(["Winter", "Summer", "Monsoon"], days)

    # REALISTIC BASE DEMAND MODEL
    base = 300 - (1.8 * price)

    season_effect = np.where(season == "Winter", 50,
                     np.where(season == "Summer", 30, 10))

    product_effect = product_effect_map[product_type] * 20

    demand = base + season_effect + product_effect

    return pd.DataFrame({
        "date": date,
        "price": price,
        "season": season,
        "demand": demand
    })

# -----------------------------
# RUN ONLY IF PRODUCT EXISTS
# -----------------------------
if product:

    df = generate_data(product_type)

    # -----------------------------
    # ENCODING
    # -----------------------------
    le = LabelEncoder()
    df["season_enc"] = le.fit_transform(df["season"])

    X = df[["price", "season_enc"]]
    y = df["demand"]

    # -----------------------------
    # MODEL
    # -----------------------------
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )
    model.fit(X, y)

    # -----------------------------
    # PREDICTION FUNCTION
    # -----------------------------
    def predict_demand(price, season):
        enc = le.transform([season])[0]
        return model.predict([[price, enc]])[0]

    # -----------------------------
    # SCENARIO COMPARISON
    # -----------------------------
    st.subheader("⚔️ Scenario Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Scenario A")
        price_A = st.slider("Price A", 50, 150, 90)
        season_A = st.selectbox("Season A", ["Winter", "Summer", "Monsoon"], key="A")

        demand_A = predict_demand(price_A, season_A)
        revenue_A = demand_A * price_A

    with col2:
        st.markdown("### Scenario B")
        price_B = st.slider("Price B", 50, 150, 120)
        season_B = st.selectbox("Season B", ["Winter", "Summer", "Monsoon"], key="B")

        demand_B = predict_demand(price_B, season_B)
        revenue_B = demand_B * price_B

    # -----------------------------
    # RESULTS
    # -----------------------------
    st.subheader("📊 Results")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Scenario A Demand", f"{demand_A:.2f}")
        st.metric("Scenario A Revenue", f"{revenue_A:.2f}")

    with col4:
        st.metric("Scenario B Demand", f"{demand_B:.2f}")
        st.metric("Scenario B Revenue", f"{revenue_B:.2f}")

    # -----------------------------
    # AI RECOMMENDATION
    # -----------------------------
    st.subheader("🤖 AI Recommendation")

    if revenue_A > revenue_B:
        st.success("👉 Scenario A is more profitable")
    else:
        st.success("👉 Scenario B is more profitable")

    if demand_A > demand_B:
        st.info("👉 Scenario A has higher demand")
    else:
        st.info("👉 Scenario B has higher demand")

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    st.subheader("🧠 Model Explanation")

    importance = model.feature_importances_

    st.write(f"Price Impact: {importance[0]:.2f}")
    st.write(f"Season Impact: {importance[1]:.2f}")

    if importance[0] > importance[1]:
        st.info("👉 Price is the dominant factor")
    else:
        st.info("👉 Season is the dominant factor")

    # -----------------------------
    # FINAL INSIGHT
    # -----------------------------
    st.subheader("📌 Final Insight")

    if revenue_A > revenue_B:
        st.success(f"For {product}: Scenario A is optimal")
    else:
        st.success(f"For {product}: Scenario B is optimal")

else:
    st.info("Enter a product to start analysis")
