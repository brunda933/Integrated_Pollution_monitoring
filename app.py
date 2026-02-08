
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models
water_model = joblib.load("water_model.pkl")
air_model = joblib.load("air_model.pkl")
soil_model = joblib.load("soil_model.pkl")

# Load scalers
water_scaler = joblib.load("water_scaler.pkl")
air_scaler = joblib.load("air_scaler.pkl")
soil_scaler = joblib.load("soil_scaler.pkl")

# Load features
water_cols = joblib.load("water_features.pkl")
air_cols = joblib.load("air_features.pkl")
soil_cols = joblib.load("soil_features.pkl")

st.set_page_config(page_title="Integrated Pollution Monitor")

st.title("🌍 Integrated Air, Water & Soil Monitoring System")

tab1, tab2, tab3 = st.tabs(["Water", "Air", "Soil"])

# ---------------- WATER ----------------
with tab1:

    st.header("💧 Water Quality")

    w_vals = []

    for c in water_cols:
        v = st.number_input(c, key="w"+c)
        w_vals.append(v)

    if st.button("Analyze Water"):

        arr = np.array(w_vals).reshape(1,-1)
        arr = water_scaler.transform(arr)

        pred = water_model.predict(arr)[0]
        prob = water_model.predict_proba(arr)[0][1]

        score = round(prob*100,2)

        if pred == 1:
            st.success("Water is SAFE")
        else:
            st.error("Water is UNSAFE")

        st.metric("Health Score", f"{score}%")
        import matplotlib.pyplot as plt

        params = water_cols
        values = w_vals

        fig, ax = plt.subplots(figsize=(10,5))

        ax.plot(params, values, marker='o', color='blue')
        ax.set_title("Water Quality Parameters")
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Values")

        plt.xticks(rotation=45)

        st.pyplot(fig)


# ---------------- AIR ----------------
with tab2:

    st.header("🌫️ Air Quality")

    a_vals = []

    for c in air_cols:
        v = st.number_input(c, key="a"+c)
        a_vals.append(v)

    if st.button("Analyze Air"):

        arr = np.array(a_vals).reshape(1,-1)
        arr = air_scaler.transform(arr)

        pred = air_model.predict(arr)[0]

        if pred == 1:
            st.success("Air is CLEAN")
        else:
            st.error("Air is POLLUTED")
            # ---------------- GRAPH ----------------
        st.subheader("📊 Air Parameter Analysis")

        air_plot_df = pd.DataFrame({
            "Parameter": air_cols,
            "Value": a_vals
        })

        st.bar_chart(
            air_plot_df.set_index("Parameter")
        )


# ---------------- SOIL ----------------
with tab3:

    st.header("🌱 Soil Quality")

    s_vals = []

    for c in soil_cols:
        v = st.number_input(c, key="s"+c)
        s_vals.append(v)

    if st.button("Analyze Soil"):

        arr = np.array(s_vals).reshape(1,-1)
        arr = soil_scaler.transform(arr)

        pred = soil_model.predict(arr)[0]

        if pred == 1:
            st.success("Soil is HEALTHY")
        else:
            st.error("Soil is CONTAMINATED")
            # ---------------- GRAPH ----------------
        st.subheader("📊 Soil Parameter Analysis")

        soil_plot_df = pd.DataFrame({
            "Parameter": soil_cols,
            "Value": s_vals
        })

        st.bar_chart(
            soil_plot_df.set_index("Parameter")
        )
