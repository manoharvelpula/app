import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Demand Intelligence Engine", layout="wide")
st.title("🚀 AI Demand & Pricing Optimization System")

# -----------------------------
# PRODUCT INPUT
# -----------------------------
st.subheader("🛍️ Product Setup")

product = st.text_input("Enter Product Name", "", placeholder="e.g., Jacket, Ice Cream")

product_type = st.selectbox(
    "Select Product Type",
    ["Winter Product", "Summer Product", "All-Season Product"]
)

season_list = ["Winter", "Summer", "Monsoon"]

# -----------------------------
# DATA GENERATION (REALISTIC)
# -----------------------------
product_effect_map = {
    "Winter Product": 1,
    "Summer Product": 1,
    "All-Season Product": 1
}

@st.cache_data
def generate_data(product_type):
    np.random.seed(42)
    days = 300

    date = pd.date_range(start="2023-01-01", periods=days)
    price = np.random.uniform(50, 150, days)
    season = np.random.choice(season_list, days)

    # REALISTIC DEMAND MODEL
    base = 300 - (1.8 * price)

    season_effect = np.where(
        season == "Winter", 50,
        np.where(season == "Summer", 30, 10)
    )

    product_effect = product_effect_map[product_type] * 20

    demand = base + season_effect + product_effect

    # NORMALIZATION (IMPORTANT FIX)
    demand = demand / 10

    return pd.DataFrame({
        "price": price,
        "season": season,
        "demand": demand
    })

# -----------------------------
# CONFIDENCE SCORE
# -----------------------------
def confidence_score(model, X_input):
    preds = np.array([
        tree.predict(X_input)[0]
        for tree in model.estimators_
    ])
    confidence = 100 - (np.std(preds) / np.mean(preds) * 100)
    return max(50, min(confidence, 99))

# -----------------------------
# PROFIT FUNCTION (FIXED)
# -----------------------------
def calculate_profit(price, demand):
    cost = price * 0.45   # realistic cost model
    return (price - cost) * demand

# -----------------------------
# PRICE OPTIMIZATION
# -----------------------------
def find_best_profit_price(season):
    prices = np.arange(50, 151, 5)
    best_profit = -1
    best_price = 0

    for p in prices:
        demand = predict_demand(p, season)
        profit = calculate_profit(p, demand)

        if profit > best_profit:
            best_profit = profit
            best_price = p

    return best_price, best_profit

# -----------------------------
# PRICING STRATEGY
# -----------------------------
def pricing_strategy(demand, price):
    elasticity = demand / price

    if elasticity > 2:
        return "📈 Penetration Pricing (Low price, high demand)"
    elif elasticity > 1:
        return "⚖️ Balanced Pricing Strategy"
    else:
        return "💎 Premium Pricing Strategy"

# -----------------------------
# RUN APP
# -----------------------------
if product:

    df = generate_data(product_type)

    le = LabelEncoder()
    df["season_enc"] = le.fit_transform(df["season"])

    X = df[["price", "season_enc"]]
    y = df["demand"]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)

    def predict_demand(price, season):
        enc = le.transform([season])[0]
        return model.predict([[price, enc]])[0]

    # -----------------------------
    # SCENARIOS
    # -----------------------------
    st.subheader("⚔️ Scenario Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Scenario A")
        price_A = st.slider("Price A", 50, 150, 90)
        season_A = st.selectbox("Season A", season_list, key="A")

        demand_A = predict_demand(price_A, season_A)
        revenue_A = price_A * demand_A
        profit_A = calculate_profit(price_A, demand_A)

    with col2:
        st.markdown("### Scenario B")
        price_B = st.slider("Price B", 50, 150, 120)
        season_B = st.selectbox("Season B", season_list, key="B")

        demand_B = predict_demand(price_B, season_B)
        revenue_B = price_B * demand_B
        profit_B = calculate_profit(price_B, demand_B)

    # -----------------------------
    # BEST SCENARIO ANALYSIS
    # -----------------------------
    st.subheader("🏆 Best Scenario Analysis")

    if demand_A > demand_B:
        st.success("👉 Scenario A has HIGHER DEMAND")
    else:
        st.success("👉 Scenario B has HIGHER DEMAND")

    if revenue_A > revenue_B:
        st.success("👉 Scenario A has HIGHER REVENUE")
    else:
        st.success("👉 Scenario B has HIGHER REVENUE")

    if profit_A > profit_B:
        st.success("👉 Scenario A has HIGHER PROFIT")
    else:
        st.success("👉 Scenario B has HIGHER PROFIT")

    # -----------------------------
    # METRICS
    # -----------------------------
    st.subheader("📊 Metrics Overview")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Demand A", f"{demand_A:.2f}")
        st.metric("Revenue A", f"{revenue_A:.2f}")
        st.metric("Profit A", f"{profit_A:.2f}")

    with col4:
        st.metric("Demand B", f"{demand_B:.2f}")
        st.metric("Revenue B", f"{revenue_B:.2f}")
        st.metric("Profit B", f"{profit_B:.2f}")

    # -----------------------------
    # SEASON PRICING ADVICE
    # -----------------------------
    st.subheader("🌦️ Season-Based Pricing Suggestion")

    if product_type == "Winter Product":
        if season_A == "Winter":
            advice = "🔥 Increase price (peak winter demand)"
        else:
            advice = "📉 Reduce price (off-season)"
    elif product_type == "Summer Product":
        if season_A == "Summer":
            advice = "🔥 Increase price (peak summer demand)"
        else:
            advice = "📉 Reduce price (off-season)"
    else:
        advice = "⚖️ Maintain balanced pricing (all-season product)"

    st.info(advice)

    # -----------------------------
    # PROFIT OPTIMIZATION
    # -----------------------------
    st.subheader("💰 Profit Optimization Mode")

    best_price, best_profit = find_best_profit_price(season_A)

    st.success(f"Best Price: ₹{best_price}")
    st.success(f"Max Profit: ₹{best_profit:.2f}")

    # -----------------------------
    # WHAT-IF ANALYSIS
    # -----------------------------
    st.subheader("🔮 What-If Analysis")

    what_if_price = st.slider("Adjust Price", 50, 150, 100)

    what_if_demand = predict_demand(what_if_price, season_A)
    what_if_profit = calculate_profit(what_if_price, what_if_demand)

    st.metric("Predicted Demand", f"{what_if_demand:.2f}")
    st.metric("Predicted Profit", f"{what_if_profit:.2f}")

    # -----------------------------
    # CONFIDENCE SCORE
    # -----------------------------
    st.subheader("📊 Confidence Score")

    X_A = [[price_A, le.transform([season_A])[0]]]
    conf_A = confidence_score(model, X_A)

    st.metric("Model Confidence", f"{conf_A:.2f}%")

else:
    st.info("Enter product name to start analysis 🚀")
