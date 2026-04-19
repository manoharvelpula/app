import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Demand Intelligence Engine", layout="wide")
st.title("🚀 Demand Forecasting System")

# -----------------------------
# MULTI-PRODUCT SUPPORT
# -----------------------------
st.subheader("🛍️ Product Setup")

products = st.text_area(
    "Enter Product Names (comma separated)",
    " apple "
)

product_list = [p.strip() for p in products.split(",") if p.strip()]

product_type = st.selectbox(
    "Select Product Seasonality",
    ["Winter Product", "Summer Product", "All-Season Product"]
)

# -----------------------------
# DATA GENERATION
# -----------------------------


@st.cache_data
def generate_data(product_type):
    np.random.seed(42)
    days = 200
    date = pd.date_range(start="2023-01-01", periods=days)

    price = np.random.uniform(50, 150, days)
    season = np.random.choice(["Winter", "Summer", "Monsoon"], days)

    base = 200 - price + np.random.normal(0, 10, days)

    season_effect = []

    for s in season:
        if product_type == "Winter Product":
            effect = 30 if s == "Winter" else -10
        elif product_type == "Summer Product":
            effect = 30 if s == "Summer" else -10
        else:
            effect = 10

        season_effect.append(effect)

    demand = base + np.array(season_effect)

    return pd.DataFrame({
        "date": date,
        "price": price,
        "season": season,
        "demand": demand
    })


if product_list:

    df = generate_data(product_type)

    # -----------------------------
    # MODEL
    # -----------------------------
    le = LabelEncoder()
    df["season_enc"] = le.fit_transform(df["season"])

    X = df[["price", "season_enc"]]
    y = df["demand"]

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X, y)

    # -----------------------------
    # SCENARIO COMPARISON
    # -----------------------------
    st.subheader("⚔️ Scenario Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Scenario A")
        price_A = st.slider("Price A", 50, 150, 90)
        season_A = st.selectbox("Season A", ["Winter", "Summer", "Monsoon"])

        enc_A = le.transform([season_A])[0]
        demand_A = model.predict([[price_A, enc_A]])[0]

    with col2:
        st.markdown("### Scenario B")
        price_B = st.slider("Price B", 50, 150, 120)
        season_B = st.selectbox("Season B", ["Winter", "Summer", "Monsoon"])

        enc_B = le.transform([season_B])[0]
        demand_B = model.predict([[price_B, enc_B]])[0]

    revenue_A = demand_A * price_A
    revenue_B = demand_B * price_B

    # -----------------------------
    # RESULTS
    # -----------------------------
    st.subheader("📊 Results")

    st.write("### Per Product Comparison")

    for product in product_list:
        st.markdown(f"#### {product}")

        col3, col4 = st.columns(2)

        with col3:
            st.metric("Demand A", f"{demand_A:.2f}")
            st.metric("Revenue A", f"{revenue_A:.2f}")

        with col4:
            st.metric("Demand B", f"{demand_B:.2f}")
            st.metric("Revenue B", f"{revenue_B:.2f}")

    # -----------------------------
    # AI RECOMMENDATION ENGINE
    # -----------------------------
    st.subheader("🤖 AI Recommendations")

    if revenue_A > revenue_B:
        st.success("👉 Scenario A is better for higher revenue")
    else:
        st.success("👉 Scenario B is better for higher revenue")

    if demand_A > demand_B:
        st.info("👉 Scenario A gives higher demand")
    else:
        st.info("👉 Scenario B gives higher demand")

    if price_A < price_B:
        st.warning("👉 Lower price tends to increase demand")
    else:
        st.warning("👉 Higher price may reduce demand")

    # -----------------------------
    # EXPLAINABLE AI
    # -----------------------------
    st.subheader("🧠 Model Explanation")

    importance = model.feature_importances_

    st.write("Feature Importance:")
    st.write(f"Price Impact: {importance[0]:.2f}")
    st.write(f"Season Impact: {importance[1]:.2f}")

    if importance[0] > importance[1]:
        st.info("👉 Price has more influence on demand")
    else:
        st.info("👉 Season has more influence on demand")

    # -----------------------------
    # FINAL INSIGHT
    # -----------------------------
    st.subheader("📌 Final Insight")

    if revenue_B > revenue_A:
        st.success("Overall: Scenario B is more profitable")
    else:
        st.success("Overall: Scenario A is more profitable")

else:
    st.info("Enter at least one product")
