import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Pricing Engine", layout="wide")
st.title("🚀 AI Demand & Pricing Optimization System (Fixed)")

# -----------------------------
# INPUT
# -----------------------------
product = st.text_input("Enter Product Name", "", placeholder="e.g., Jacket, Ice Cream, Apple")

product_type = st.selectbox(
    "Select Product Type",
    ["Winter Product", "Summer Product", "All-Season Product"]
)

season_list = ["Winter", "Summer", "Monsoon"]

# -----------------------------
# REALISTIC DEMAND FUNCTION (CORE FIX)
# -----------------------------
def true_demand(price, season, product_type):

    base = 250

    # realistic price elasticity curve (NOT linear)
    price_effect = 300 / (1 + np.exp(0.05 * (price - 100)))

    season_map = {
        "Winter": 1.3,
        "Summer": 1.2,
        "Monsoon": 1.0
    }

    product_map = {
        "Winter Product": 1.2,
        "Summer Product": 1.1,
        "All-Season Product": 1.0
    }

    return base + price_effect * season_map[season] * product_map[product_type]


# -----------------------------
# DATA GENERATION (REALISTIC)
# -----------------------------
@st.cache_data
def generate_data(product_type):

    np.random.seed(42)
    n = 300

    price = np.random.randint(50, 150, n)
    season = np.random.choice(season_list, n)

    demand = np.array([
        true_demand(price[i], season[i], product_type)
        for i in range(n)
    ])

    # small noise for ML realism
    demand = demand + np.random.normal(0, 5, n)

    df = pd.DataFrame({
        "price": price,
        "season": season,
        "demand": demand
    })

    return df


# -----------------------------
# CONFIDENCE SCORE
# -----------------------------
def confidence_score(model, X_input):
    preds = np.array([tree.predict(X_input)[0] for tree in model.estimators_])
    score = 100 - (np.std(preds) / np.mean(preds) * 100)
    return max(60, min(score, 99))


# -----------------------------
# PROFIT OPTIMIZATION
# -----------------------------
def best_profit_price(season):
    best_price = 0
    best_profit = -1

    for p in range(50, 151, 5):
        d = true_demand(p, season, product_type)
        profit = (p - 30) * d

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
# MAIN APP
# -----------------------------
if product:

    df = generate_data(product_type)

    # encoding
    le = LabelEncoder()
    df["season_enc"] = le.fit_transform(df["season"])

    X = df[["price", "season_enc"]]
    y = df["demand"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    # predictions
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

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
        revenue_A = demand_A * price_A

    with col2:
        st.markdown("### Scenario B")
        price_B = st.slider("Price B", 50, 150, 120)
        season_B = st.selectbox("Season B", season_list, key="B")

        demand_B = predict_demand(price_B, season_B)
        revenue_B = demand_B * price_B

    # -----------------------------
    # RESULTS
    # -----------------------------
    st.subheader("📊 Results")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Demand A", f"{demand_A:.2f}")
        st.metric("Revenue A", f"{revenue_A:.2f}")

    with col4:
        st.metric("Demand B", f"{demand_B:.2f}")
        st.metric("Revenue B", f"{revenue_B:.2f}")

    # -----------------------------
    # AI DECISION
    # -----------------------------
    st.subheader("🤖 AI Recommendation")

    best = "A" if revenue_A > revenue_B else "B"
    st.success(f"Scenario {best} is best for revenue")

    # -----------------------------
    # STRATEGY
    # -----------------------------
    st.subheader("🎯 Pricing Strategy")

    st.info(pricing_strategy(demand_A, price_A))

    # -----------------------------
    # WHAT-IF
    # -----------------------------
    st.subheader("🔮 What-If Analysis")

    what_price = st.slider("Adjust Price", 50, 150, 100)

    what_demand = predict_demand(what_price, season_A)
    what_revenue = what_price * what_demand

    st.metric("Demand", f"{what_demand:.2f}")
    st.metric("Revenue", f"{what_revenue:.2f}")

    # -----------------------------
    # PROFIT OPTIMIZATION
    # -----------------------------
    st.subheader("💰 Profit Optimization")

    bp, bp_profit = best_profit_price(season_A)

    st.success(f"Best Price: ₹{bp}")
    st.success(f"Max Profit: ₹{bp_profit:.2f}")

    # -----------------------------
    # CONFIDENCE
    # -----------------------------
    st.subheader("📊 Confidence Score")

    X_A = [[price_A, le.transform([season_A])[0]]]
    conf = confidence_score(model, X_A)

    st.metric("Confidence", f"{conf:.2f}%")

    # -----------------------------
    # MODEL ACCURACY
    # -----------------------------
    st.subheader("📏 Model Accuracy")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("R² Score", f"{r2:.2f}")

    with col2:
        st.metric("MAE", f"{mae:.2f}")

else:
    st.info("Enter product to start 🚀")
